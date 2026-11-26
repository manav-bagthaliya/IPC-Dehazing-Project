
from basicsr.utils import scandir
from basicsr.utils.lmdb_util import make_lmdb_from_imgs


def prepare_keys_div2k(folder_path):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=False)))
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     '--dataset',
    #     type=str,
    #     help=("Options: 'DIV2K', 'REDS', 'Vimeo90K' "
    #           'You may need to modify the corresponding configurations in codes.'))
    # args = parser.parse_args()
    # dataset = args.dataset.lower()
    # if dataset == 'div2k':
    #     create_lmdb_for_div2k()
    # elif dataset == 'reds':
    #     create_lmdb_for_reds()
    # elif dataset == 'vimeo90k':
    #     create_lmdb_for_vimeo90k()
    # else:
    #     raise ValueError('Wrong dataset.')

        # HR images
    folder_path = 'datasets/HQ_sub'
    lmdb_path = 'datasets/HQ_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
