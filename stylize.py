from utils import getContentByMask, get_prefix, get_config, check_folder, mkdir_output_test, get_file_list, \
    create_html_tabel
from data_processing.data_processing import load_data_testing, save_images
import argparse, os, random, json, shutil
from inference import Inference
from glob import glob
import numpy as np
from scipy import misc

parses = argparse.ArgumentParser()
parses.add_argument('--config', type=str, default='configs/test.yaml', help='Path to the configs file.')
opts = parses.parse_args()


def predict_test(inference, dir_result, dir_style, dir_content, img_size, data_process_type):
    list_path_content = get_file_list(dir_content)
    list_path_style = get_file_list(dir_style)

    dir_out_img = os.path.join(dir_result, 'image')
    dir_out_content = os.path.join(dir_result, 'content')
    dir_out_style = os.path.join(dir_result, 'style')
    dir_out_index = os.path.join(dir_result, 'index')

    check_folder(dir_out_img)
    check_folder(dir_out_content)
    check_folder(dir_out_style)
    check_folder(dir_out_index)

    # write html for visual comparison
    index0 = create_html_tabel(os.path.join(dir_out_index, 'index0.html'), ['name', 'style'])
    w, h = 256, 256
    count = 0
    sum_time = 0
    count_file = len(list_path_content) * len(list_path_style)
    for style_file in list_path_style:
        style_prefix = get_prefix(style_file)
        Is = load_data_testing(style_file, img_size[1], data_process_type[1])

        out_style_path = os.path.join(dir_out_style, '{}.jpg'.format(style_prefix))
        save_images(Is, [1, 1], out_style_path)

        path_index_style = os.path.join(dir_out_index, style_prefix + '_index0.html')
        index = create_html_tabel(path_index_style, ['name', 'content', 'stylizedImage', 'style'])

        for content_file in list_path_content:
            content_prefix = get_prefix(content_file)

            Ic = load_data_testing(content_file, img_size[0], data_process_type[0])
            results, time = inference.predict(Ic, Is)
            Ics = results['Ics']

            print("Processing: size_content: (%d,%d)   size_style: (%d,%d)" % (
                Ic.shape[1], Ic.shape[2], Is.shape[1], Is.shape[2]))

            count += 1
            if count > 50:
                sum_time += time

            out_content_path = os.path.join(dir_out_content, '{}.jpg'.format(content_prefix))
            save_images(Ic, [1, 1], out_content_path)

            image_path = os.path.join(dir_out_img, '{}-{}.jpg'.format(style_prefix, content_prefix))
            save_images(Ics, [1, 1], image_path)

            index.write("<td>%s</td>" % (style_prefix + '-' + content_prefix))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (out_content_path, w, h))

            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (
            '../image' + os.path.sep + os.path.basename(image_path), w, h))

            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (out_style_path, w, h))
            index.write("</tr>")
        index.close()
        index0.write("<td><a href=%s >%s</a></td>" % (path_index_style, os.path.basename(style_file)))
        index0.write("<td><img src='%s' width='%d' height='%d'></td>" % (out_style_path, w, h))
        index0.write("</tr>")
    index0.close()

    if count_file > 50:
        avg_time = sum_time / (count_file - 50)
        print('average stylized in:', avg_time)


def predict_test_random(inference, dir_result, dir_s, dir_c, random_count, random_size, img_size, data_process_type):
    list_path_content = glob(dir_c + '/*.*')
    list_path_style = glob(dir_s + '/*.*')
    w, h = 256, 256

    for idx0 in range(0, random_count):
        dir_out_img = os.path.join(dir_result, str(idx0), 'image')
        dir_out_content = os.path.join(dir_result, str(idx0), 'content')
        dir_out_style = os.path.join(dir_result, str(idx0), 'style')
        dir_out_index = os.path.join(dir_result, str(idx0))
        check_folder(dir_out_img)
        check_folder(dir_out_content)
        check_folder(dir_out_style)
        check_folder(dir_out_index)

        path_index = os.path.join(dir_out_index, 'index0.html')
        index = create_html_tabel(path_index, ['name', 'content', 'stylizedImage', 'style'])

        list_path_in_content = random.sample(list_path_content, random_size)
        list_path_in_style = random.sample(list_path_style, random_size)

        for content_file, style_file in zip(list_path_in_content, list_path_in_style):
            Is = load_data_testing(style_file, img_size[1], data_process_type[1])
            Ic = load_data_testing(content_file, img_size[0], data_process_type[0])
            results, _ = inference.predict(Ic, Is)
            Ics = results['Ics']

            print("Processing: size_content: (%d,%d)   size_style: (%d,%d)" % (
            Ic.shape[1], Ic.shape[2], Is.shape[1], Is.shape[2]))

            style_prefix = get_prefix(style_file)
            content_prefix = get_prefix(content_file)

            image_path = os.path.join(dir_out_img, '{}-{}.jpg'.format(style_prefix, content_prefix))
            save_images(Ics, [1, 1], image_path)

            path_out_content = os.path.join(dir_out_content, os.path.basename(content_file))
            save_images(Ic, [1, 1], path_out_content)

            path_out_style = os.path.join(dir_out_style, os.path.basename(style_file))
            save_images(Is, [1, 1], path_out_style)

            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (
            'content/' + os.path.basename(path_out_content), w, h))
            index.write(
                "<td><img src='%s' width='%d' height='%d'></td>" % ('image/' + os.path.basename(image_path), w, h))
            index.write(
                "<td><img src='%s' width='%d' height='%d'></td>" % ('style/' + os.path.basename(path_out_style), w, h))
            index.write("</tr>")

        index.close()



def main():
    args = get_config(opts.config)
    if args is None:
        exit()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['GPU_ID'][0])
    phase = args['phase']
    dir_s = args['data']['dir_style']
    dir_c = args['data']['dir_content']
    data_process_type = args['data_process_type']
    img_size_c = args['img_size_c']
    img_size_s = args['img_size_s']
    img_size = [img_size_c, img_size_s]

    dir_result = mkdir_output_test(args)
    inference = Inference(args)

    if phase == 'test':
        predict_test(inference, dir_result, dir_s, dir_c, img_size, data_process_type)

    elif phase == 'test_random':
        random_count = args['random_count']
        random_size = args['random_size']
        predict_test_random(inference, dir_result, dir_s, dir_c, random_count, random_size, img_size, data_process_type)


if __name__ == '__main__':
    main()