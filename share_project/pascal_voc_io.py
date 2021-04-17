import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
from pathlib import Path

XML_EXT = '.xml'
ENCODE_METHOD = "utf-8"


class PascalVocWriter:

    def __init__(self, foldername, filename, img_size, databaseSrc="Unknown", local_img_path=None):
        self.foldername = foldername
        self.filename = filename  # with suffix
        self.databaseSrc = databaseSrc
        self.img_size = img_size  # height, width, channels
        self.local_img_path = local_img_path
        self.bbox_list = []
        self.verified = False

    def _gen_xml(self):
        if self.filename is None or \
                self.foldername is None or \
                self.img_size is None:
            return None

        top = Element("annotation")
        if self.verified:
            top.set("verified", "yes")

        folder = SubElement(top, "folder")
        folder.text = self.foldername

        filename = SubElement(top, "filename")
        filename.text = self.filename

        if self.local_img_path is not None:
            local_img_path = SubElement(top, "path")
            local_img_path.text = self.local_img_path

        source = SubElement(top, "source")
        database = SubElement(source, "database")
        database.text = self.databaseSrc

        size_part = SubElement(top, "size")
        width = SubElement(size_part, "width")
        height = SubElement(size_part, "height")
        depth = SubElement(size_part, "depth")
        width.text = str(self.img_size[1])
        height.text = str(self.img_size[0])
        if len(self.img_size) == 3:
            depth.text = str(self.img_size[2])
        else:
            depth.text = '1'

        segmented = SubElement(top, "segmented")
        segmented.text = '0'
        return top

    def _prettify(self, elem):
        rought_string = ET.tostring(elem, encoding="utf-8")
        root = etree.fromstring(rought_string)
        return etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace(
            "  ".encode(), "\t".encode())

    def add_bbox(self, name, difficult, bbox):
        bndbox = {'xmin': bbox[0], 'ymin': bbox[1],
                  'xmax': bbox[2], 'ymax': bbox[3]}
        bndbox['name'] = name
        bndbox['difficult'] = difficult
        self.bbox_list.append(bndbox)

    def _append_objects(self, top):
        for each_object in self.bbox_list:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            name.text = str(each_object['name'])
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            truncated = SubElement(object_item, 'truncated')
            if int(float(each_object['ymax'])) == int(
                    float(self.img_size[0])) or (int(float(each_object['ymin'])) == 1):
                truncated.text = "1"  # max == height or min
            elif (int(float(each_object['xmax'])) == int(float(self.img_size[1]))) or (int(float(each_object['xmin'])) == 1):
                truncated.text = "1"  # max == width or min
            else:
                truncated.text = "0"
            difficult = SubElement(object_item, 'difficult')
            difficult.text = str(bool(each_object['difficult']) & 1)
            bndbox = SubElement(object_item, 'bndbox')
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])

    def save(self, target_file=None):
        root = self._gen_xml()
        self._append_objects(root)
        out_file = None
        if target_file is None:
            out_file = codecs.open(
                self.filename + XML_EXT, 'w', encoding=ENCODE_METHOD)
        else:
            out_file = codecs.open(target_file, 'w', encoding=ENCODE_METHOD)

        prettify_result = self._prettify(root)
        out_file.write(prettify_result.decode('utf8'))
        out_file.close()


class PascalVocReader:

    def __init__(self, filepath):
        # shapes type:
        # [label, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        if isinstance(filepath, str):
            self.filepath = Path(filepath)
        else:
            self.filepath = filepath
        self.verified = False
        try:
            self.parse_xml()
        except:
            pass

    def get_shapes(self):
        return self.shapes

    def add_shape(self, label, bndbox, difficult):
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        # points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        points = (xmin, ymin, xmax, ymax)
        # self.shapes.append((label, points, None, None, difficult))
        self.shapes.append(
            {
                'name': label,
                'bbox': points,
                'difficult': difficult
            }
        )

    def parse_xml(self):
        assert self.filepath.suffix == XML_EXT, "Unsupport file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xmltree = ET.parse(self.filepath, parser=parser).getroot()
        filename = xmltree.find('filename').text
        try:
            verified = xmltree.attrib['verified']
            if verified == 'yes':
                self.verified = True
        except KeyError:
            self.verified = False

        for object_iter in xmltree.findall('object'):
            bndbox = object_iter.find("bndbox")
            label = object_iter.find('name').text
            # Add chris
            difficult = False
            if object_iter.find('difficult') is not None:
                difficult = bool(int(object_iter.find('difficult').text))
            self.add_shape(label, bndbox, difficult)
        return True

if __name__ == "__main__":
    xml_writer = PascalVocWriter("Dongba", "P0001.png", (1024, 1024, 3))
    xml_writer.add_bbox("garbage", False, (100, 100, 300, 500))
    xml_writer.save()
