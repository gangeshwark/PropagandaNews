__author__ = "Giovanni Da San Martino"
__copyright__ = "Copyright 2019"
__credits__ = ["Giovanni Da San Martino"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Giovanni Da San Martino"
__email__ = "gmartino@hbku.edu.qa"
__status__ = "Beta"

import argparse
import codecs

import src.article_annotations as an
import src.propaganda_techniques as pt


def main(args):
    span_file = args.spans_file
    article_file = args.article_file
    print_line_numbers = bool(args.add_line_numbers)

    an.techniques = pt.Propaganda_Techniques(filename="data/propaganda-techniques-names-semeval2020task11.txt")
    annotations = an.Articles_annotations()

    annotations.load_article_annotations_from_csv_file(span_file)

    with codecs.open(article_file, "r", encoding="utf8") as f:
        article_content = f.read()

    output_text, footnotes, legend = annotations.mark_text(article_content, print_line_numbers)

    print(output_text)
    print(legend)
    print(footnotes)


def main1(args):
    span_file = args.spans_file
    article_file = args.article_file

    with codecs.open(article_file, "r", encoding="utf8") as f:
        article_content = f.read()
    with codecs.open(span_file, "r", encoding="utf8") as f:
        span_content = f.read()

    spans = span_content.split('\n')
    # print(article_content)
    # print(spans)
    for span in spans:
        if span:
            id_, label, start, end = span.split('\t')
            print(label, start, end, article_content[int(start):int(end)])
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Highlight labelled spans in a text file. Do not pipe with less. \n" +
                                                 "Example: print_spans.py -s data/article736757214.task-FLC.labels -t data/article736757214.txt")
    parser.add_argument('-t', '--text-file', dest='article_file', required=True, help="file with text document")
    parser.add_argument('-s', '--spans-file', dest='spans_file', required=True,
                        help="file with spans to be highlighted. One line of the span file")
    parser.add_argument('-l', '--add-line-numbers', dest='add_line_numbers', required=False,
                        action='store_true', help="Prepend line numbers on output.")
    main1(parser.parse_args())
    main(parser.parse_args())
