
import outpainting
from outpainting import *


def test(image, imageName):
    src_file = image
    dst_file = './static/Generated_Image/' + imageName.split('/')[-1]
    gen_model = load_model('./generator_final.pt')
    # print('Source file: ' + src_file + '...')
    input_img = plt.imread(src_file)[:, :, :]
    output_img, blended_img = perform_outpaint(gen_model, input_img)
    # plt.imsave(dst_file,output_img)
    plt.imsave(dst_file, blended_img)
    print('Destination file: ' + dst_file + ' written')