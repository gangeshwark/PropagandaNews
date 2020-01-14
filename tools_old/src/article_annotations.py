#from __future__ import annotations
import sys
import src.annotation as ans
import src.annotation_w_o_label as anwol
from src.propaganda_techniques import Propaganda_Techniques
import logging.handlers

__author__ = "Giovanni Da San Martino"
__copyright__ = "Copyright 2019"
__credits__ = ["Giovanni Da San Martino"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Giovanni Da San Martino"
__email__ = "gmartino@hbku.edu.qa"
__status__ = "Beta"

logger = logging.getLogger("propaganda_scorer")

techniques = Propaganda_Techniques()

class Articles_annotations(object):

    """
    Class for handling annotations for one article. 
    Articles_annotations is composed of an article id
    and a list of Annotation objects. 
    """

    start_annotation_effect = "\033[42;33m"
    end_annotation_effect = "\033[m"
    start_annotation_str = "{"
    end_annotation_str = "}"
    annotation_background_color = "\033[44;33m"


    def __init__(self, spans:anwol.AnnotationWithOutLabel=None, article_id=None):

        if spans is None:
            self.spans = []
        else:
            self.spans = spans
        self.article_id = article_id


    def __len__(self):

        return len(self.spans)


    def __str__(self):

        return "article id: %s\n%s"%(self.article_id, "\n".join(self.spans))


    def add_annotation(self, annotation, article_id:str=None):
        """
        param annotation: an Annotation object
        """
        if article_id is None:
            article_id = self.get_article_id()
        self.add_article_id(article_id)
        #if not isinstance(annotation, Annotation):
        #    sys.exit()
        self.spans.append(annotation)


    def add_article_id(self, article_id):

        if self.article_id is None:
            self.article_id = article_id
        else:
            if article_id is not None and self.article_id != article_id:
                logger.error("Trying to add an annotation with a different article id")
                sys.exit()


    def get_article_id(self):

        return self.article_id


    def get_article_annotations(self):

        return self.spans


    def get_markers_from_spans(self):

        self.sort_spans()
        self.markers = []
        for i, annotation in enumerate(self.spans, 1):
            self.markers.append((annotation.get_start_offset(), annotation.get_label(), i, "start"))
            self.markers.append((annotation.get_end_offset(), annotation.get_label(), i, "end"))
        self.markers = sorted(self.markers, key=lambda ann: ann[0])


    def groupby_technique(self):

        annotation_list = {}
        for i, annotation in enumerate(self.get_article_annotations()):
            technique = annotation.get_label()
            if technique not in annotation_list.keys():
                annotation_list[technique] = []
            annotation_list[technique].insert(0, i)
        return annotation_list


    #check_annotation_spans_with_category_matching
    def has_overlapping_spans(self, prop_vs_non_propaganda, merge_overlapping_spans=False):
        """
        Check whether there are ovelapping spans for the same technique in the same article.
        Two spans are overlapping if their associated techniques match (according to category_matching_func)
        If merge_overlapping_spans==True then the overlapping spans are merged, otherwise an error is raised.

        :param merge_overlapping_spans: if True merges the overlapping spans
        :return:
        """

        annotation_list = {}
        for annotation in self.get_article_annotations():
            if prop_vs_non_propaganda:
                technique = "propaganda"
            else:
                technique = annotation.get_label()
            if technique not in annotation_list.keys():
                annotation_list[technique] = [annotation] #[[technique, curr_span]]
            else:
                if merge_overlapping_spans:
                    annotation_list[technique].append(annotation)
                    self.merge_article_annotations(annotation_list[technique], len(annotation_list[technique]) - 1)
                else:
                    for matching_annotation in annotation_list[technique]:
                        if annotation.span_overlapping(matching_annotation):
                            logger.error("In article %s, the span of the annotation %s, [%s,%s] overlap with "
                                         "the following one from the same article:%s, [%s,%s]" % (
                                             self.get_article_id(), annotation.get_label(),
                                             annotation.get_start_offset(), annotation.get_end_offset(), matching_annotation.get_label(), matching_annotation.get_start_offset(), matching_annotation.get_end_offset()))
                            return False
                    annotation_list[technique].append([annotation])
        if merge_overlapping_spans: # recreate the list of annotations
            self.reset_annotations()
            for anlist in annotation_list.values():
                for a in anlist: 
                    self.add_annotation(a)
        return True


    def is_starting_marker(self, marker_index=None):

        if marker_index is None:
            marker_index = self.curr_marker
        if marker_index < len(self.markers):
            return self.markers[marker_index][3] == "start"


    def is_ending_marker(self, marker_index=None):

        if marker_index is None:
            marker_index = self.curr_marker
        if marker_index < len(self.markers):
            return self.markers[marker_index][3] == "end"


    def load_article_annotations_from_csv_file(self, filename):
        """
        Read annotations from a csv file and creates a list of
        Annotation objects. Check class annotation for details
        on the file format.
        """
        with open(filename, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                an, article_id = ans.Annotation.load_annotation_from_string(line.rstrip(), i, filename)
                self.add_annotation(an, article_id)


    def marker_label(self, marker_index=None):

        if marker_index is None:
            marker_index = self.curr_marker
        if marker_index < len(self.markers):
            return self.markers[marker_index][1]
        # else:
        # ERROR


    def marker_position(self, marker_index=None):

        if marker_index is None:
            marker_index = self.curr_marker
        if marker_index < len(self.markers):
            return self.markers[marker_index][0]


    def marker_annotation(self, marker_index=None):

        if marker_index is None:
            marker_index = self.curr_marker
        if marker_index < len(self.markers):
            return self.markers[marker_index][2]


    def mark_text(self, original_text, print_line_numbers=False):
        """
        mark the string original_text with object's annotations

        original_text: string with the text to be marked
        print_line_numbers: add line numbers to the text

        :return output_text the text in string original_text with added marks
                footnotes the list of techniques in the text
                legend description of the marks added
        """

        self.get_markers_from_spans()

        output_text, curr_output_text_index, self.curr_marker = ("", 0, 0)
        footnotes = "List of techniques found in the article\n\n"
        techniques_found = set()
        annotations_stack = []  # to handle overlapping annotations when assigning color background
        while curr_output_text_index < len(original_text):
            if self.curr_marker >= len(self.markers):
                output_text += original_text[curr_output_text_index:]
                curr_output_text_index = len(original_text)
            else:
                if self.marker_position() <= curr_output_text_index:
                    if self.is_starting_marker():
                        output_text += self.start_annotation_effect + self.start_annotation_str
                        annotations_stack.append(self.marker_annotation())
                    else:
                        output_text += "%s%s%s" % (
                            self.end_annotation_effect, "" if len(annotations_stack) > 1 else " ",
                            self.start_annotation_effect)
                    techniques_index = techniques.indexOf(self.marker_label())
                    output_text += str(techniques_index)
                    techniques_found.add(techniques_index)
                    if self.is_ending_marker():
                        output_text += self.end_annotation_str + self.end_annotation_effect
                        annotations_stack.remove(self.marker_annotation())
                        if len(annotations_stack) > 0:
                            output_text += self.annotation_background_color
                    else:
                        output_text += self.end_annotation_effect + " " + self.annotation_background_color
                    self.curr_marker += 1
                else:
                    output_text += original_text[curr_output_text_index:self.marker_position()]
                    curr_output_text_index = self.marker_position()

        if print_line_numbers:
            indices, char_index = ([], 0)
            for line in original_text.split("\n"):
                indices.append(char_index)
                char_index += len(line) + 1
            #output_text = "\n".join(["%d (%d) %s"%(i, x[0], x[1])
            output_text = "\n".join(["%d %s"%(i, x[1]) 
                                     for i, x in enumerate(zip(indices, output_text.split("\n")), 1)])

        legend = "---\n%sHighlighted text%s: any propagandistic fragment\n%s%si%s: start of the i-th technique" \
                 "\n%si%s%s: end of the i-th technque\n---"\
                 %(self.annotation_background_color, self.end_annotation_effect, self.start_annotation_effect,
                   self.start_annotation_str, self.end_annotation_effect, self.start_annotation_effect,
                   self.end_annotation_str, self.end_annotation_effect)

        for technique_index in sorted(techniques_found):
            footnotes += "%d: %s\n" % (technique_index, techniques[technique_index])

        return output_text, footnotes, legend


    def add_sentence_marker(self, line:str, row_counter:int)->str:

        if max(line.find("<span"), line.find("</span")) > -1: # there is an annotation in this row
            return '<div class="technique" id="row%d">%s</div>\n'%(row_counter, line)
        else:
            if len(line) <= 1: #empty line
                return '<br/>'
            else:   
                return '<div>%s</div>\n'%(line)


    def annotation_stack_index_to_markers_index(self, ind:int)->int:

        for x in range(len(self.markers)):
            if self.marker_annotation(x)==ind:
                return x
        sys.exit()


    def technique_index_from_annotation_index(self, x:int)->int:

        return techniques.indexOf(self.marker_label(self.annotation_stack_index_to_markers_index(x)))


    def start_annotation_marker_function(self, annotations_stack:list, marker_index:int, row_counter:int)->str:

        return '<span id="row%dannotation%d" class="%s">' \
                %(row_counter, self.marker_annotation(marker_index), " ".join([ "technique%d"%(self.technique_index_from_annotation_index(x)) for x in annotations_stack + [ self.marker_annotation(marker_index) ] ]))
        #if len(annotations_stack) > 0: # there is at least another tag opened, this one will overlap with it
        #    return '<span id="row%dannotation%d" class="%s overlappingtechniques">' \
        #        %(row_counter, self.marker_annotation(marker_index), " ".join([ "technique%d"%(self.technique_index_from_annotation_index(x)) for x in annotations_stack + [ self.marker_annotation(marker_index) ] ]))
        #else:
        #    return '<span id="row%dannotation%d" class="technique%d technique">'%(row_counter, self.marker_annotation(marker_index), techniques.indexOf(self.marker_label(marker_index))) 


    def end_annotation_marker_function(self, annotations_stack:list, marker_index:int, row_counter:int)->str:

        if self.marker_annotation() != annotations_stack[-1]: # we are facing this case: <t1> <t2> </t1> </t2> and we are about to close </t1> (self.marker_annotation()==</t1>, annotations_stack[-1]==</t2>), however, that case above is not allowed in HTML, therefore we are about to transform it to <t1> <t2> </t2></t1><t2> </t2> below
            new_annotations_stack = annotations_stack[ annotations_stack.index(self.marker_annotation()): ]
            res = "".join([ "</span>" for x in new_annotations_stack ]) # closing all tags opened that are supposed to continue after </t2>
            new_annotations_stack.remove(self.marker_annotation()) # removing </t1> from annotations_stack copy 
            technique_index = techniques.indexOf(self.marker_label(marker_index))
            res += '<sup id="row%dannotation%d" class="technique%d">%d</sup>'%(row_counter, self.marker_annotation(), technique_index, technique_index)
            for x in new_annotations_stack:
                new_annotations_stack.remove(x) # self.start_annotation_marker_function() assumes x is not in the annotations_stack variable passed as parameter
                res += self.start_annotation_marker_function(new_annotations_stack, self.annotation_stack_index_to_markers_index(x), row_counter) 
            return res
        else: # end of non-overlapping technique
            technique_index = techniques.indexOf(self.marker_label(marker_index))
            return '</span><sup id="row%dannotation%d" class="technique%d">%d</sup>'%(row_counter, self.marker_annotation(), technique_index, technique_index) 


    def tag_text_with_annotations(self, original_text, print_line_numbers=False):
        """
        mark the string original_text with object's annotations

        original_text: string with the text to be marked
        print_line_numbers: add line numbers to the text

        :return output_text the text in string original_text with added marks
                footnotes the list of techniques in the text
                legend description of the marks added
        """

        self.get_markers_from_spans()

        output_text, curr_output_text_index, self.curr_marker = ("", 0, 0)
        techniques_found = set()
        row_counter = 1
        #print(self.markers)
        annotations_stack = []  # to handle overlapping annotations when assigning color background
        while curr_output_text_index < len(original_text):
            if self.curr_marker >= len(self.markers): # done marking text, need to flush the remaining content of <original_text> into <output_text>
                output_text += original_text[curr_output_text_index:]
                curr_output_text_index = len(original_text)
            else: # more markers have to be added to the content string
                if self.marker_position() <= curr_output_text_index: # it is time to add a marker
                    techniques_index = techniques.indexOf(self.marker_label())
                    techniques_found.add(techniques_index)
                    if self.is_starting_marker():
                        output_text += self.start_annotation_marker_function(annotations_stack, self.curr_marker, row_counter)
                        annotations_stack.append(self.marker_annotation())
                    else: 
                        output_text += self.end_annotation_marker_function(annotations_stack, self.curr_marker, row_counter)
                        annotations_stack.remove(self.marker_annotation())
                    self.curr_marker += 1
                else: # flush string content up to the next marker
                    text_to_be_added = original_text[curr_output_text_index:self.marker_position()]
                    row_counter += text_to_be_added.count('\n')
                    output_text += text_to_be_added
                    curr_output_text_index = self.marker_position()

        final_text = ""
        for row_counter, line in enumerate(output_text.split("\n"), 1):
            final_text += self.add_sentence_marker(line, row_counter)

        footnotes = "\n<div>List of techniques found in the article</div>\n\n"
        for technique_index in sorted(techniques_found):
            footnotes += "<div>%d: %s</div>\n" % (technique_index, techniques[technique_index])

        return final_text, footnotes


    def merge_article_annotations(self, annotations_without_overlapping, i):
        """
        Checks if annotations_without_overlapping
        :param annotations_without_overlapping: a list of Annotations objects of an article assumed to be
                without overlapping
        :param i: the index in spans which needs to be tested for overlapping
        :return: 
        """
        #print("checking element %d of %d"%(i, len(spans)))
        if i<0:
            return True
        for j in range(0, i): #len(annotations_without_overlapping)):
            assert i<len(annotations_without_overlapping) or print(i, len(annotations_without_overlapping))
            if j != i and annotations_without_overlapping[i].span_overlapping(annotations_without_overlapping[j]):
                #   len(annotations_without_overlapping[i][1].intersection(annotations_without_overlapping[j][1])) > 0:
                # print("Found overlapping spans: %d-%d and %d-%d in annotations %d,%d:\n%s"
                #       %(min(annotations_without_overlapping[i][1]), max(annotations_without_overlapping[i][1]),
                #         min(annotations_without_overlapping[j][1]), max(annotations_without_overlapping[j][1]), i,j,
                #         print_annotations(annotations_without_overlapping)))
                annotations_without_overlapping[j].merge_spans(annotations_without_overlapping[i])
                #annotations_without_overlapping[j][1] = annotations_without_overlapping[j][1].union(annotations_without_overlapping[i][1])
                del(annotations_without_overlapping[i])
                # print("Annotations after deletion:\n%s"%(print_annotations(annotations_without_overlapping)))
                if j > i:
                    j -= 1
                # print("calling recursively")
                self.merge_article_annotations(annotations_without_overlapping, j)
                # print("done")
                return True

        return False


    def remove_empty_annotations(self):

        self.spans = [ span for span in self.spans if span is not None ]


    def set_output_format(self, article_id=True, span=True, label=True):
        """
        Defines which fields are printed when annotations are written to standard output or file
        """
        self.output_format_article_id = article_id
        self.output_format_article_spans = span
        self.output_format_article_label = label


    def annotations_to_string_csv(self):
        """
        write article annotations, one per line, in the following format:
        article_id  label   span_start  span_end
        """
        span_string=""
        for span in self.spans:
            span_data = []
            if self.output_format_article_id:
                span_data.append(self.get_article_id())
            if self.output_format_article_label:
                span_data.append(span.get_label())
            if self.output_format_article_spans:
                span_data.append("%d\t%d"%(span.get_start_offset(), span.get_end_offset()))
            span_string += "\t".join(span_data) + "\n"

        return span_string


    def reset_annotations(self):

        self.spans = []


    @classmethod
    def set_start_annotation_effect(cls, new_effect:str)->None:

        cls.start_annotation_effect = new_effect


    @classmethod
    def set_end_annotation_effect(cls, new_effect:str)->None:

        cls.end_annotation_effect = new_effect


    @classmethod
    def set_start_annotation_str(cls, new_effect:str)->None:

        cls.start_annotation_str = new_effect


    @classmethod
    def set_end_annotation_str(cls, new_effect:str)->None:

        cls.end_annotation_str = new_effect


    @classmethod
    def set_annotation_background_color(cls, new_effect:str)->None:

        cls.annotation_background_color = new_effect


    def save_annotations_to_file(self, filename):
        
        with open(filename, "w") as f:
            f.write(self.annotations_to_string_csv())


    def shift_spans(self, start_index, offset):

        for span in self.spans:
            if span.get_start_offset() >= start_index:
                span.shift_annotation(offset)


    def sort_spans(self):
        """
        sort the list of annotations with respect to the starting offset
        """
        self.spans = sorted(self.spans, key=lambda span: span.get_start_offset() )

