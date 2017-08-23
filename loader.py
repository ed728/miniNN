from struct import unpack
import sys

MAGIC = {
            2049: 'LABELS',
            2051: 'IMAGES'
        }
def parse_magic(magic):
    return MAGIC.get(magic)

def load_header(data):
    if len(data) < 8:
        raise ValueError("Data smaller than header size.")
    header = dict()
    magic, size = unpack('>ii', data[0:8])
    header['MAGIC'] = magic
    header['TYPE'] = parse_magic(magic)
    header['DATA_SIZE'] = size
    header['HEADER_SIZE'] = 8
    if header['TYPE'] == 'IMAGES':
        if len(data) < 16:
            raise ValueError("Data smaller than header size.")
        size_x, size_y = unpack('>ii', data[8:16])
        header['ROWS'] = size_x
        header['COLUMNS'] = size_y
        header['HEADER_SIZE'] = 16
    return header


def load_labels(data, sz):
    labels = unpack('B'*sz, data) # Don't care about endiness as we are loading bytes.
    return labels

def load_images_flat(data, sz, x, y):
    """
    This function returns a list of images. Each image is just a flat list of pixel values.
    """
    # x is amount of rows
    # y is amount of columns
    imgs = []
    img_sz = x*y
    for i in range(sz):
        # I believe it is more efficient to unpack everything at the same time, and then slice it into images,
        # but it was done this way to keep the code simple, and we don't really care about the small speed-up.
        start = i * img_sz
        end = start + img_sz
        imgs.append(unpack('B' * img_sz, data[start:end]))
    return imgs

def load_NN(path):
    f = open(path, 'rb')
    data = f.read()
    header = load_header(data)
    data = data[header['HEADER_SIZE']]
    if header['TYPE'] == 'LABELS':
        labels = load_labels(data, header['DATA_SIZE'])
        N = max(labels)
        if N == 0:
            raise ValueError("Bad data: 0 is highest label.") # Not an impossible case, but not interesting for us, so just error out.
        Y = list()
        for l in labels:
            tmp = [0] * N
            tmp[l] = 1
            Y.append(tmp)
        return Y
    elif header['TYPE'] == 'IMAGES':
        images = load_images_flat(data, header['DATA_SIZE'], hader['ROWS'], header['COLUMNS'])
        return images

if __name__=='__main__':
    path = sys.argv[1]
    f = open(path, 'rb')
    data = f.read()
    header = load_header(data)
    print(header)
    data = data[header['HEADER_SIZE']:]
    if header['TYPE'] == 'LABELS':
        labels = load_labels(data, header['DATA_SIZE'])
        print(len(labels))
        print(labels[0:10])
    elif header['TYPE'] == 'IMAGES':
        images = load_images_flat(data, header['DATA_SIZE'], header['ROWS'], header['COLUMNS'])
        print(len(images))
        fmt = "{} \t" * (header['COLUMNS'])
        for i in range(header['ROWS']):
            start = i * header['COLUMNS']
            end = start + header['COLUMNS']
            print("\t".join([str(x) for x in images[0][start:end]]))
            print
