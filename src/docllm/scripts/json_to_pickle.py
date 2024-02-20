import os
import sys

from docllm.data.preprocessing.doc_data_to_pickle import document_tokenization_from_file


def main(input_file_or_dir: str, output_dir: str, use_page_dimensions: bool) -> None:
    if os.path.isfile(input_file_or_dir):
        input_files = [input_file_or_dir]
    else:
        input_files = [
            os.path.join(input_file_or_dir, fn) for fn in os.listdir(input_file_or_dir) if fn.endswith(".json")
        ]
    for filename in input_files:
        document_tokenization_from_file(filename, output_dir, use_page_dimensions)


if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) < 3:
        print("Usage: python document_tokenization.py <input_file_or_dir> <output_dir> [<use_page_dimensions>]")
        print("Default value for use_page_dimensions is True.")
        sys.exit(1)
    input_file_or_dir = sys.argv[1]
    output_dir = sys.argv[2]
    use_page_dimensions = bool(sys.argv[3]) if len(sys.argv) > 3 else True
    main(input_file_or_dir, output_dir, use_page_dimensions)
