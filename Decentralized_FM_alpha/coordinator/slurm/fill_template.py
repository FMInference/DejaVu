import argparse

import os
import sys

def main():
    # Create the parser
    my_parser = argparse.ArgumentParser(description='List the content of a folder')

    # Add the arguments
    my_parser.add_argument('--infer-data', type=str, required=True, help='')
    my_parser.add_argument('--template-path', type=str, required=True, help='')
    my_parser.add_argument('--output-path', type=str, required=True, help='')

    # Execute the parse_args() method
    args = my_parser.parse_args()

    with open(args.template_path) as f:
        text = f.read()
        text = text.replace('{{infer_data}}', args.infer_data)

    with open(args.output_path, 'w') as f:
        f.write(text)

if __name__ == '__main__':
    main()
