import json
import os
from pathlib import Path
from collections import Counter, defaultdict
import random
from typing import List, Dict, Set


class FEVERDatasetReducer:
    """
    Reduce FEVER dataset size while maintaining class balance and filtering relevant evidence.
    """

    def __init__(self,
                 train_size: int = 10000,
                 dev_size: int = 2000,
                 test_size: int = 2000,
                 output_dir: str = "reduced_fever_data",
                 random_seed: int = 42,
                 additional_random_docs: int = 10000):
        """
        Initialize the dataset reducer.

        Args:
            train_size: Number of samples for training set
            dev_size: Number of samples for dev set
            test_size: Number of samples for test set
            output_dir: Directory to save reduced datasets
            random_seed: Random seed for reproducibility
            additional_random_docs: Number of random distractor documents to include
        """
        self.train_size = train_size
        self.dev_size = dev_size
        self.test_size = test_size
        self.output_dir = output_dir
        self.random_seed = random_seed
        self.additional_random_docs = additional_random_docs
        self.count_evidence = 0

        random.seed(random_seed)

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Track which evidence pages are needed
        self.required_evidence_ids = set()

    def load_jsonl(self, filepath: str, max_lines: int = None) -> List[Dict]:
        """Load JSONL file."""
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_lines and i >= max_lines:
                        break
                    data.append(json.loads(line.strip()))
            print(f"✓ Loaded {len(data)} records from {filepath}")
        except FileNotFoundError:
            print(f"❌ File not found: {filepath}")
        return data

    def save_jsonl(self, data: List[Dict], filepath: str):
        """Save data to JSONL file."""
        output_path = os.path.join(self.output_dir, filepath)
        with open(output_path, 'a', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print(f"✓ Saved {len(data)} records to {output_path}")

    def get_label_distribution(self, data: List[Dict]) -> Dict[str, int]:
        """Get distribution of labels in the dataset."""
        labels = [item['label'] for item in data if 'label' in item]
        return dict(Counter(labels))

    def stratified_sample(self, data: List[Dict], sample_size: int) -> List[Dict]:
        """
        Perform stratified sampling to maintain class balance.

        Args:
            data: List of data samples
            sample_size: Desired sample size

        Returns:
            Stratified sample maintaining class proportions
        """
        # Group by label
        label_groups = defaultdict(list)
        for item in data:
            label = item.get('label', 'UNKNOWN')
            label_groups[label].append(item)

        # Calculate original distribution
        total_samples = len(data)
        label_distribution = {label: len(samples) / total_samples
                              for label, samples in label_groups.items()}

        print(f"\nOriginal distribution ({total_samples} samples):")
        for label, prop in sorted(label_distribution.items()):
            count = len(label_groups[label])
            print(f"  {label}: {count} ({prop * 100:.2f}%)")

        # Sample proportionally from each class
        sampled_data = []
        for label, samples in label_groups.items():
            # Calculate how many samples needed from this class
            n_samples = int(sample_size * label_distribution[label])

            # Ensure we don't sample more than available
            n_samples = min(n_samples, len(samples))

            # Random sample
            sampled = random.sample(samples, n_samples)
            sampled_data.extend(sampled)

        # If we're short due to rounding, add random samples to reach target
        if len(sampled_data) < sample_size:
            remaining = sample_size - len(sampled_data)
            all_remaining = [item for item in data if item not in sampled_data]
            if all_remaining:
                sampled_data.extend(random.sample(all_remaining,
                                                  min(remaining, len(all_remaining))))

        # Shuffle the final sample
        random.shuffle(sampled_data)

        print(f"\nSampled distribution ({len(sampled_data)} samples):")
        sampled_dist = self.get_label_distribution(sampled_data)
        for label, count in sorted(sampled_dist.items()):
            prop = count / len(sampled_data)
            print(f"  {label}: {count} ({prop * 100:.2f}%)")

        return sampled_data

    def extract_evidence_ids(self, data: List[Dict]):
        """Extract all evidence document IDs from the dataset."""
        print("\nExtracting evidence IDs from claims...")
        evidence_count = 0

        for item in data:
            if 'evidence' in item and item['evidence']:
                for evidence_set in item['evidence']:
                    if evidence_set:  # Check if not None
                        for evidence in evidence_set:
                            if evidence and len(evidence) >= 3:
                                doc_id = evidence[2]
                                if doc_id and doc_id is not None:
                                    self.required_evidence_ids.add(doc_id.strip().lower())
                                    evidence_count += 1

        print(f"✓ Found {evidence_count} evidence references")
        print(f"✓ Extracted {len(self.required_evidence_ids)} unique document IDs")

    def filter_all_evidence_files(self, evidence_filepaths: List[str],
                                  output_filename: str = "filtered_evidence.jsonl"):
        """
        Filter ALL evidence files to only include documents referenced in the reduced dataset,
        plus additional random documents as distractors (globally, not per file).

        Args:
            evidence_filepaths: List of paths to all evidence files
            output_filename: Name for the filtered evidence file
        """
        print(f"\nFiltering {len(evidence_filepaths)} evidence files...")
        print(f"Looking for {len(self.required_evidence_ids)} unique documents...")

        filtered_evidence = []
        random_candidates = []
        total_docs = 0
        non_required_count = 0

        # Process ALL evidence files
        for evidence_filepath in evidence_filepaths:
            print(f"  Reading: {os.path.basename(evidence_filepath)}")
            try:
                with open(evidence_filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        total_docs += 1
                        doc = json.loads(line.strip())

                        # Check if this document is needed
                        doc_id = doc.get('id', '').strip().lower()

                        if doc_id in self.required_evidence_ids:
                            filtered_evidence.append(doc)
                        else:
                            non_required_count += 1
                            # Reservoir sampling to keep memory usage bounded
                            if len(random_candidates) < self.additional_random_docs:
                                random_candidates.append(doc)
                            else:
                                # Random replacement with decreasing probability
                                replace_idx = random.randint(0, non_required_count - 1)
                                if replace_idx < self.additional_random_docs:
                                    random_candidates[replace_idx] = doc

                        if total_docs % 100000 == 0:
                            print(f"    Processed {total_docs} documents, found {len(filtered_evidence)} relevant...")

            except FileNotFoundError:
                print(f"  ❌ File not found: {evidence_filepath}")

        print(f"\n✓ Processed {total_docs} total documents across all files")
        print(f"✓ Found {len(filtered_evidence)} relevant documents")

        # Add random distractor documents (exactly self.additional_random_docs)
        if self.additional_random_docs > 0 and random_candidates:
            num_to_add = min(self.additional_random_docs, len(random_candidates))
            filtered_evidence.extend(random_candidates[:num_to_add])
            print(f"✓ Added {num_to_add} random distractor documents")

        # Shuffle the final evidence set
        random.shuffle(filtered_evidence)

        self.save_jsonl(filtered_evidence, output_filename)

        coverage = (len(filtered_evidence) - num_to_add) / len(self.required_evidence_ids) * 100
        print(f"✓ Coverage: {coverage:.2f}% of required evidence found")
        print(f"✓ Total documents in filtered evidence: {len(filtered_evidence)}")
        print(f"  - Required evidence: {len(filtered_evidence) - num_to_add}")
        print(f"  - Random distractors: {num_to_add}")


    def process_dataset(self,
                        train_file: str,
                        dev_file: str,
                        test_file: str,
                        evidence_files: List[str] = None):
        """
        Process all dataset files and create reduced versions.

        Args:
            train_file: Path to training file
            dev_file: Path to dev file
            test_file: Path to test file
            evidence_files: List of paths to evidence files (optional)
        """
        print("=" * 70)
        print("FEVER Dataset Reduction")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Train size: {self.train_size}")
        print(f"  Dev size: {self.dev_size}")
        print(f"  Test size: {self.test_size}")
        print(f"  Additional random docs: {self.additional_random_docs}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Random seed: {self.random_seed}")

        # Process training data
        print("\n" + "=" * 70)
        print("Processing Training Data")
        print("=" * 70)
        train_data = self.load_jsonl(train_file)
        if train_data:
            reduced_train = self.stratified_sample(train_data, self.train_size)
            self.save_jsonl(reduced_train, "train.jsonl")
            self.extract_evidence_ids(reduced_train)

        # Process dev data
        print("\n" + "=" * 70)
        print("Processing Dev Data")
        print("=" * 70)
        dev_data = self.load_jsonl(dev_file)
        if dev_data:
            reduced_dev = self.stratified_sample(dev_data, self.dev_size)
            self.save_jsonl(reduced_dev, "paper_dev.jsonl")
            self.extract_evidence_ids(reduced_dev)

        # Process test data
        print("\n" + "=" * 70)
        print("Processing Test Data")
        print("=" * 70)
        test_data = self.load_jsonl(test_file)
        if test_data:
            reduced_test = self.stratified_sample(test_data, self.test_size)
            self.save_jsonl(reduced_test, "paper_test.jsonl")
            self.extract_evidence_ids(reduced_test)

            # Process evidence files
        if evidence_files:
            print("\n" + "=" * 70)
            print("Processing Evidence Files")
            print("=" * 70)

            # Clear output file if it exists
            output_path = os.path.join(self.output_dir, "wiki/filtered_evidence.jsonl")
            if os.path.exists(output_path):
                os.remove(output_path)
                print(f"✓ Cleared existing evidence file")

            # Process all evidence files at once
            self.filter_all_evidence_files(evidence_files)

        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"✓ Reduced datasets saved to: {self.output_dir}/")
        print(f"✓ Total unique evidence documents needed: {len(self.required_evidence_ids)}")


def get_all_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        root = str(root)
        for filename in files:
            filename = str(filename)
            full_path = os.path.join(root, filename)
            file_paths.append(full_path)
    return file_paths


if __name__ == "__main__":
    BASE_DIR = "../../../../dataset/"

    # Configure your desired sizes here
    reducer = FEVERDatasetReducer(
        train_size=10000,
        dev_size=2000,
        test_size=2000,
        output_dir=BASE_DIR + "reduced_fever_data",
        random_seed=42,
        additional_random_docs=50000
    )

    evidence = get_all_file_paths(BASE_DIR + "wiki-pages/wiki-pages")

    # Process the datasets
    reducer.process_dataset(
        train_file=BASE_DIR + "train.jsonl",
        dev_file=BASE_DIR + "paper_dev.jsonl",
        test_file=BASE_DIR + "paper_test.jsonl",
        evidence_files=evidence
    )

    print("\n✓ Done! You can now load the reduced datasets from the 'reduced_fever_data' folder.")