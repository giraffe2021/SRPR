import json
import os

from tqdm import tqdm

from tools.augmentations import *

try:
    from .mini_imagenet_name import data_name_dict
except:
    from mini_imagenet_name import data_name_dict
resize_size = 92
center_crop_size = 84
e_dim = 640


# resize_size = 224
# center_crop_size = 224
# dim = 768

@tf.function
def load_embedding_block(z_path, padding=2048):
    # 读取并解析张量
    z = tf.io.parse_tensor(tf.io.read_file(z_path), tf.float32)
    # 获取原始长度
    original_length = tf.shape(z)[0]
    # 补齐操作
    z = tf.pad(z, [[0, padding - original_length], [0, 0]])
    # 生成mask，标记有效数据
    mask = tf.sequence_mask([original_length], maxlen=padding, dtype=tf.float32)
    mask = tf.squeeze(mask, axis=0)  # 调整mask的形状以匹配z的第一维
    # 实际有效长度
    valid_length = original_length
    return z, mask, valid_length


@tf.function
def load_embedding(z_path):
    x = tf.io.parse_tensor(tf.io.read_file(z_path), tf.float32)
    return x


@tf.function
def process_embedding(path, way_num, episode_test_sample_num):
    x = tf.map_fn(load_embedding, path, dtype=tf.float32, parallel_iterations=16)
    x.set_shape([path.shape[0], e_dim])
    x = tf.stack(tf.split(x, way_num), axis=0)

    query_x = x[:, :episode_test_sample_num, ...]
    support_x = x[:, episode_test_sample_num:, ...]

    return support_x, query_x


@tf.function
def process_embedding_blocks(path, way_num, padding=2048):
    z, mask, valid_length = tf.map_fn(partial(load_embedding_block, padding=padding), path,
                                      dtype=(tf.float32, tf.float32, tf.int32),
                                      parallel_iterations=16)
    z = tf.concat(tf.split(z, way_num), axis=0)
    mask = tf.concat(tf.split(mask, way_num), axis=0)
    return z, mask, valid_length


@tf.function
def read_image(path):
    image = tf.io.read_file(path)  # 根据路径读取图片
    image = tf.image.decode_jpeg(image, channels=3)  # 图片解码
    image = tf.cast(image, dtype=tf.float32) / 255.
    image = tf.image.resize(image, [resize_size, resize_size])
    image = tf.image.central_crop(image, center_crop_size / resize_size)
    image = tf.image.resize(image, [center_crop_size, center_crop_size])
    # image = image[...,::-1]
    return image


@tf.function
def process_with_augment(path, meta_label, global_label, way_num, episode_test_sample_num,
                         global_label_depth):
    images = tf.map_fn(read_image_with_random_crop_resize, path, parallel_iterations=16, dtype=tf.float32)
    images = tf.map_fn(do_augmentations, images, parallel_iterations=16)
    # images = (images - tf.cast([0.485, 0.456, 0.406], tf.float32)) / tf.cast([0.229, 0.224, 0.225], tf.float32)
    meta_label = tf.one_hot(meta_label, axis=-1, depth=way_num)
    global_label = tf.one_hot(global_label, axis=-1, depth=global_label_depth)
    return split(images, meta_label, global_label, way_num, episode_test_sample_num)


@tf.function
def process_with_mip_up_augment(path, meta_label, global_label, way_num, episode_test_sample_num,
                                global_label_depth):
    images = tf.map_fn(read_image_with_random_crop_resize, path, parallel_iterations=16, dtype=tf.float32)
    x = tf.map_fn(do_augmentations, images, parallel_iterations=16)
    # x = (x - tf.cast([0.485, 0.456, 0.406], tf.float32)) / tf.cast([0.229, 0.224, 0.225], tf.float32)

    meta_y = tf.one_hot(meta_label, axis=-1, depth=way_num)
    y = tf.one_hot(global_label, axis=-1, depth=global_label_depth)

    alfa_shape = tf.unstack(tf.shape(images)[:1])
    alfa = tf.random.uniform(alfa_shape, 0.3, 1.0)
    alfa_x = tf.reshape(alfa, [*alfa_shape, 1, 1, 1])

    indices = tf.random.shuffle(tf.range(tf.shape(x)[0]))
    x_reference = tf.gather(x, indices)
    y_reference = tf.gather(y, indices)
    meta_y_reference = tf.gather(meta_y, indices)
    # 保持 相同类别,和不同类别的混合
    x = x * alfa_x + x_reference * (1.0 - alfa_x)
    alfa_y = tf.reshape(alfa, [*alfa_shape, 1])
    y = y * alfa_y + y_reference * (1.0 - alfa_y)
    meta_y = meta_y * alfa_y + meta_y_reference * (1.0 - alfa_y)

    return split(x, meta_y, y, way_num, episode_test_sample_num)


@tf.function
def process_without_augment(path, meta_label, global_label, way_num, episode_test_sample_num,
                            global_label_depth):
    images = tf.map_fn(read_image, path, parallel_iterations=16, dtype=tf.float32)

    # images = (images - tf.cast([0.485, 0.456, 0.406], tf.float32)) / tf.cast([0.229, 0.224, 0.225], tf.float32)

    meta_label = tf.one_hot(meta_label, axis=-1, depth=way_num)
    global_label = tf.one_hot(global_label, axis=-1, depth=global_label_depth)
    return split(images, meta_label, global_label, way_num, episode_test_sample_num)


@tf.function
def read_image_with_random_crop_resize(path):
    image = tf.io.read_file(path)  # 根据路径读取图片
    image = tf.image.decode_jpeg(image, channels=3)  # 图片解码

    image = tf.cast(image, dtype=tf.float32) / 255.
    image = random_resize_and_crop(image, (center_crop_size, center_crop_size))
    return image


@tf.function
def split(x, y, g_y, way_num, episode_test_sample_num):
    x = tf.stack(tf.split(x, way_num), axis=0)
    y = tf.stack(tf.split(y, way_num), axis=0)
    g_y = tf.stack(tf.split(g_y, way_num), axis=0)

    query_x = x[:, :episode_test_sample_num, ...]
    query_y = y[:, :episode_test_sample_num, ...]
    query_g_y = g_y[:, :episode_test_sample_num, ...]

    support_x = x[:, episode_test_sample_num:, ...]
    surport_y = y[:, episode_test_sample_num:, ...]
    surport_g_y = g_y[:, episode_test_sample_num:, ...]

    return (support_x, surport_y, surport_g_y), (query_x, query_y, query_g_y)


@tf.function
def process_with_crop(path, global_label,
                      global_label_depth):
    image = tf.io.read_file(path)  # 根据路径读取图片
    image = tf.image.decode_jpeg(image, channels=3)  # 图片解码
    image = tf.cast(image, dtype=tf.float32) / 255.
    image = random_resize_and_crop(image)
    image = do_augmentations(image)
    global_label = tf.one_hot(global_label, axis=-1, depth=global_label_depth)
    return image, global_label


@tf.function
def process_with_crop_with_contrastive(path, global_label,
                                       global_label_depth, p=0.5):
    image = tf.io.read_file(path)  # 根据路径读取图片
    image = tf.image.decode_jpeg(image, channels=3)  # 图片解码
    image = tf.cast(image, dtype=tf.float32) / 255.
    image = random_resize_and_crop(image, ratio_min=0.8, ratio_max=4.)

    chance = tf.random.uniform([], 0, 100, dtype=tf.int32)

    @tf.function
    def process(x):
        x = horizontal_randomFlip(tf.expand_dims(x, 0))[0]
        x = color_jitter(x)
        x = color_drop(x)
        return x

    x = tf.case([(tf.less(chance, tf.cast(p * 100, dtype=tf.int32)),
                  lambda: process(image))],
                default=lambda: image)
    x_rotated_1, rotate_label_1 = rotation_process_with_limit(x)
    x_rotated_2, rotate_label_2 = rotation_process_with_limit(x)
    x_resized_1, ratio_1 = ratio_process(x, ratio_min=0.25, ratio_max=2.25)
    x_resized_2, ratio_2 = ratio_process(x, ratio_min=0.25, ratio_max=2.25)

    global_label = tf.one_hot(global_label, axis=-1, depth=global_label_depth)
    return (image, x_rotated_1, x_rotated_2, x_resized_1, x_resized_2), (
        global_label, rotate_label_1, rotate_label_2, ratio_1, ratio_2)


class MiniImageNetDataLoader:
    def __init__(self, data_dir_path, way_num=5, shot_num=1, episode_test_sample_num=15, cache=True,
                 embedding_dir_path=None):
        if not isinstance(data_dir_path, list):
            self.base_dir = data_dir_path
            self.embedding_dir_path = embedding_dir_path
            self.embedding_block_dir_path = embedding_dir_path + "_block"
            self.embedding_clusters_dir_path = embedding_dir_path + "_cluster"
            self.meta_test_folder = os.path.join(self.base_dir, "test")
            self.meta_val_folder = os.path.join(self.base_dir, "val")
            self.meta_train_folder = os.path.join(self.base_dir, "train")
            assert os.path.exists(self.meta_test_folder)
            assert os.path.exists(self.meta_val_folder)
            assert os.path.exists(self.meta_train_folder)

            info_json_path = os.path.join(self.base_dir, "info.json")

            if os.path.exists(info_json_path) and cache:
                with open(info_json_path, "r") as f:
                    all_info_dict = json.load(f)
                print("~~~~!!!!!!loading an all_info_dict from {}".format(info_json_path))

                self.meta_train_folder_lists = all_info_dict.get("meta_train_folder_lists")
                self.meta_val_folder_lists = all_info_dict.get("meta_val_folder_lists")
                self.meta_test_folder_lists = all_info_dict.get("meta_test_folder_lists")
                self.meta_train_image_dict = all_info_dict.get("meta_train_image_dict")
                self.meta_val_image_dict = all_info_dict.get("meta_val_image_dict")
                self.meta_test_image_dict = all_info_dict.get("meta_test_image_dict")
                self.meta_test_global_label_dict = all_info_dict.get("meta_test_global_label_dict")
                self.meta_val_global_label_dict = all_info_dict.get("meta_val_global_label_dict")
                self.meta_train_global_label_dict = all_info_dict.get("meta_train_global_label_dict")
                self.num_images_train = all_info_dict.get("num_images_train")
                self.num_images_val = all_info_dict.get("num_images_val")
                self.num_images_test = all_info_dict.get("num_images_test")

                last_basedir = all_info_dict.get("base_dir")
                if last_basedir is not None:
                    self.meta_train_folder_lists = [_.replace(last_basedir, self.base_dir) for _ in
                                                    self.meta_train_folder_lists]
                    self.meta_val_folder_lists = [_.replace(last_basedir, self.base_dir) for _ in
                                                  self.meta_val_folder_lists]
                    self.meta_test_folder_lists = [_.replace(last_basedir, self.base_dir) for _ in
                                                   self.meta_test_folder_lists]

                    self.meta_train_image_dict = {
                        k.replace(last_basedir, self.base_dir): [_.replace(last_basedir, self.base_dir) for _ in v] for
                        k, v
                        in self.meta_train_image_dict.items()}

                    self.meta_val_image_dict = {
                        k.replace(last_basedir, self.base_dir): [_.replace(last_basedir, self.base_dir) for _ in v] for
                        k, v
                        in self.meta_val_image_dict.items()}

                    self.meta_test_image_dict = {
                        k.replace(last_basedir, self.base_dir): [_.replace(last_basedir, self.base_dir) for _ in v] for
                        k, v
                        in self.meta_test_image_dict.items()}

                    self.meta_train_global_label_dict = {
                        k.replace(last_basedir, self.base_dir): v for k, v
                        in self.meta_train_global_label_dict.items()}

                    self.meta_val_global_label_dict = {
                        k.replace(last_basedir, self.base_dir): v for k, v
                        in self.meta_val_global_label_dict.items()}

                    self.meta_test_global_label_dict = {
                        k.replace(last_basedir, self.base_dir): v for k, v
                        in self.meta_test_global_label_dict.items()}
            else:
                self.meta_test_folder_lists = sorted([os.path.join(self.meta_test_folder, label) \
                                                      for label in os.listdir(self.meta_test_folder) \
                                                      if os.path.isdir(os.path.join(self.meta_test_folder, label)) \
                                                      ])
                self.meta_test_global_label_dict = {foldername: index for index, foldername in
                                                    enumerate(self.meta_test_folder_lists)}

                self.meta_test_image_dict = {foldername: os.listdir(foldername) for index, foldername in
                                             enumerate(self.meta_test_folder_lists)}
                for k, v in self.meta_test_image_dict.items():
                    fileList = []
                    for root, dirs, files in os.walk(k, True):
                        for name in files:
                            path = os.path.join(root, name)
                            fileList.append(path)
                    self.meta_test_image_dict[k] = fileList

                self.num_images_test = 0
                for v in self.meta_test_image_dict.values():
                    self.num_images_test += len(v)

                self.meta_val_folder_lists = sorted([os.path.join(self.meta_val_folder, label) \
                                                     for label in os.listdir(self.meta_val_folder) \
                                                     if os.path.isdir(os.path.join(self.meta_val_folder, label)) \
                                                     ])
                self.meta_val_global_label_dict = {foldername: index for index, foldername in
                                                   enumerate(self.meta_val_folder_lists)}
                self.meta_val_image_dict = {foldername: os.listdir(foldername) for index, foldername in
                                            enumerate(self.meta_val_folder_lists)}
                for k, v in self.meta_val_image_dict.items():
                    fileList = []
                    for root, dirs, files in os.walk(k, True):
                        for name in files:
                            path = os.path.join(root, name)
                            fileList.append(path)
                    self.meta_val_image_dict[k] = fileList

                self.num_images_val = 0
                for v in self.meta_val_image_dict.values():
                    self.num_images_val += len(v)

                self.meta_train_folder_lists = sorted([os.path.join(self.meta_train_folder, label) \
                                                       for label in os.listdir(self.meta_train_folder) \
                                                       if os.path.isdir(os.path.join(self.meta_train_folder, label)) \
                                                       ])
                self.meta_train_global_label_dict = {foldername: index for index, foldername in
                                                     enumerate(self.meta_train_folder_lists)}
                self.meta_train_image_dict = {foldername: os.listdir(foldername) for index, foldername in
                                              enumerate(self.meta_train_folder_lists)}

                for k, v in self.meta_train_image_dict.items():
                    fileList = []
                    for root, dirs, files in os.walk(k, True):
                        for name in files:
                            path = os.path.join(root, name)
                            fileList.append(path)
                    self.meta_train_image_dict[k] = fileList

                self.num_images_train = 0
                for v in self.meta_train_image_dict.values():
                    self.num_images_train += len(v)

                all_info_dict = {
                    "base_dir": self.base_dir,
                    "meta_test_folder": self.meta_test_folder,
                    "meta_val_folder": self.meta_val_folder,
                    "meta_train_folder": self.meta_train_folder,
                    "meta_train_folder_lists": self.meta_train_folder_lists,
                    "meta_val_folder_lists": self.meta_val_folder_lists,
                    "meta_test_folder_lists": self.meta_test_folder_lists,
                    "meta_train_image_dict": self.meta_train_image_dict,
                    "meta_val_image_dict": self.meta_val_image_dict,
                    "meta_test_image_dict": self.meta_test_image_dict,
                    "meta_test_global_label_dict": self.meta_test_global_label_dict,
                    "meta_val_global_label_dict": self.meta_val_global_label_dict,
                    "meta_train_global_label_dict": self.meta_train_global_label_dict,
                    "num_images_train": self.num_images_train,
                    "num_images_val": self.num_images_val,
                    "num_images_test": self.num_images_test,

                }
                info_json_path = os.path.join(self.base_dir, "info.json")

                print("~~~~!!!!!!writing all_info_dict into {}".format(info_json_path))
                with open(info_json_path, "w") as f:
                    json.dump(all_info_dict, f)
        elif isinstance(data_dir_path, list) and cache is True:

            self.meta_test_folder = []
            self.meta_val_folder = []
            self.meta_train_folder = []
            self.meta_train_folder_lists = []
            self.meta_val_folder_lists = []
            self.meta_test_folder_lists = []
            self.meta_train_image_dict = {}
            self.meta_val_image_dict = {}
            self.meta_test_image_dict = {}
            self.meta_test_global_label_dict = {}
            self.meta_val_global_label_dict = {}
            self.meta_train_global_label_dict = {}
            self.num_images_train = 0
            self.num_images_val = 0
            self.num_images_test = 0
            for sub_dir in data_dir_path:
                sub_dataset = MiniImageNetDataLoader(sub_dir)
                self.meta_test_folder = self.meta_test_folder + [sub_dataset.meta_test_folder]
                self.meta_val_folder = self.meta_val_folder + [sub_dataset.meta_val_folder]
                self.meta_train_folder = self.meta_train_folder + [sub_dataset.meta_train_folder]
                self.meta_train_folder_lists = self.meta_train_folder_lists + sub_dataset.meta_train_folder_lists
                self.meta_val_folder_lists = self.meta_val_folder_lists + sub_dataset.meta_val_folder_lists
                self.meta_test_folder_lists = self.meta_test_folder_lists + sub_dataset.meta_test_folder_lists
                self.meta_train_image_dict.update(sub_dataset.meta_train_image_dict)
                self.meta_val_image_dict.update(sub_dataset.meta_val_image_dict)
                self.meta_test_image_dict.update(sub_dataset.meta_test_image_dict)
                self.meta_test_global_label_dict.update(sub_dataset.meta_test_global_label_dict)
                self.meta_val_global_label_dict.update(sub_dataset.meta_val_global_label_dict)
                self.meta_train_global_label_dict.update(sub_dataset.meta_train_global_label_dict)
                self.num_images_train = self.num_images_train + sub_dataset.num_images_train
                self.num_images_val = self.num_images_val + sub_dataset.num_images_val
                self.num_images_test = self.num_images_test + sub_dataset.num_images_test

        print("=> {} loaded".format(data_dir_path))
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # cats | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(len(self.meta_train_folder_lists), self.num_images_train))
        print("  val      | {:5d} | {:8d}".format(len(self.meta_val_folder_lists), self.num_images_val))
        print("  test     | {:5d} | {:8d}".format(len(self.meta_test_folder_lists), self.num_images_test))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(len(self.meta_train_folder_lists)
                                                  + len(self.meta_val_folder_lists)
                                                  + len(self.meta_test_folder_lists),
                                                  self.num_images_train
                                                  + self.num_images_val
                                                  + self.num_images_test))
        print("  ------------------------------")

    def get_images(self, paths, labels, nb_samples=None, shuffle=True):
        if nb_samples is not None:
            sampler = lambda x: random.sample(x, nb_samples)
        else:
            sampler = lambda x: x
        images = [(i, os.path.join(path, image)) \
                  for i, path in zip(labels, paths) \
                  for image in sampler(os.listdir(path))]
        if shuffle:
            random.shuffle(images)
        return images

    def generate_origin_data_list(self, phase='train'):
        if phase == 'train':
            folders = self.meta_train_folder_lists
            label_dict = self.meta_train_global_label_dict
            image_dict = self.meta_train_image_dict
            global_label_depth = len(label_dict)
        elif phase == 'val':
            folders = self.meta_val_folder_lists
            label_dict = self.meta_val_global_label_dict
            image_dict = self.meta_val_image_dict
            global_label_depth = len(label_dict)
        elif phase == 'test':
            folders = self.meta_test_folder_lists
            label_dict = self.meta_test_global_label_dict
            image_dict = self.meta_test_image_dict
            global_label_depth = len(label_dict)

        else:
            print('Please select vaild phase')
            all_labels = []
            all_filenames = []
            return all_filenames, all_labels
        try:
            name_projector = {v: data_name_dict[os.path.basename(k)] for k, v in label_dict.items()}
        except:
            name_projector = {}

        print('Generating filenames')

        all_filenames = []
        class_id = []
        for folderName, folderImageList in image_dict.items():
            all_filenames.extend(folderImageList)
            id = label_dict[folderName]
            class_id.extend([id for _ in range(len(folderImageList))])

        return all_filenames, class_id, global_label_depth

    def generate_data_list(self, way_num, num_samples_per_class, episode_num=None, phase='train',
                           with_embeddings=False):
        if phase == 'train':
            folders = self.meta_train_folder_lists
            label_dict = self.meta_train_global_label_dict
            image_dict = self.meta_train_image_dict
            global_label_depth = len(label_dict)

            if episode_num is None:
                episode_num = 20000
            print('Generating train filenames')

        elif phase == 'val':
            folders = self.meta_val_folder_lists
            label_dict = self.meta_val_global_label_dict
            image_dict = self.meta_val_image_dict
            global_label_depth = len(label_dict)

            if episode_num is None:
                episode_num = 600
            print('Generating val filenames')

        elif phase == 'test':
            folders = self.meta_test_folder_lists
            label_dict = self.meta_test_global_label_dict
            image_dict = self.meta_test_image_dict
            global_label_depth = len(label_dict)

            if episode_num is None:
                episode_num = 600
            print('Generating test filenames')
        else:
            print('Please select vaild phase')
            all_labels = []
            all_filenames = []
            return all_filenames, all_labels
        try:
            name_projector = {v: data_name_dict[os.path.basename(k)] for k, v in label_dict.items()}
        except:
            name_projector = {}
        all_filenames = [[] for _ in range(episode_num)]
        all_meta_labels = [[] for _ in range(episode_num)]
        all_global_labels = [[] for _ in range(episode_num)]
        if with_embeddings:
            all_embedding_filenames = [[] for _ in range(episode_num)]
            all_embedding_cluster_filenames = [[] for _ in range(episode_num)]
            all_embedding_block_filenames = [[] for _ in range(episode_num)]
        for index in tqdm(range(episode_num)):
            trycount = 1000
            for try_i in range(trycount):
                sampled_character_folders = random.sample(folders, way_num)
                random.shuffle(sampled_character_folders)
                filenames = []
                embedding_filenames = []
                labels = []
                global_labels = []
                embedding_blocks = []
                embedding_clusters = []
                if try_i < trycount - 1:
                    try:
                        for meta_label, cls in enumerate(sampled_character_folders):
                            image_list = image_dict[cls]
                            global_label = label_dict[cls]
                            images = random.sample(image_list, num_samples_per_class)
                            filenames.extend([os.path.join(cls, name) for name in images])
                            labels.extend([meta_label for _ in images])
                            global_labels.extend([global_label for _ in images])
                            if with_embeddings:
                                embedding_blocks.append(
                                    os.path.join(cls.replace(self.base_dir, self.embedding_block_dir_path), "block"))
                                embedding_clusters.append(
                                    os.path.join(cls.replace(self.base_dir, self.embedding_clusters_dir_path),
                                                 "cluster"))
                                embedding_filenames = [f.replace(self.base_dir, self.embedding_dir_path) + ".z"
                                                       for f in filenames]
                        break
                    except:
                        continue
                else:
                    for meta_label, cls in enumerate(sampled_character_folders):
                        image_list = image_dict[cls]
                        global_label = label_dict[cls]
                        images = random.sample(image_list, num_samples_per_class)
                        filenames.extend([os.path.join(cls, name) for name in images])
                        labels.extend([meta_label for _ in images])
                        global_labels.extend([global_label for _ in images])
                        if with_embeddings:
                            embedding_blocks.append(
                                os.path.join(cls.replace(self.base_dir, self.embedding_block_dir_path), "block"))
                            embedding_clusters.append(
                                os.path.join(cls.replace(self.base_dir, self.embedding_clusters_dir_path),
                                             "cluster"))
                            embedding_filenames = [f.replace(self.base_dir, self.embedding_dir_path) + ".z"
                                                   for f in filenames]

            all_filenames[index] = filenames
            all_meta_labels[index] = labels
            all_global_labels[index] = global_labels
            if with_embeddings:
                all_embedding_filenames[index] = embedding_filenames
                all_embedding_block_filenames[index] = embedding_blocks
                all_embedding_cluster_filenames[index] = embedding_clusters
        if with_embeddings:
            return (all_filenames, all_meta_labels, all_global_labels), global_label_depth, name_projector, (
                all_embedding_filenames, all_embedding_block_filenames, all_embedding_cluster_filenames)

        return (all_filenames, all_meta_labels, all_global_labels), global_label_depth, name_projector

    def get_dataset(self, phase='train', way_num=5, shot_num=5, episode_test_sample_num=15, episode_num=None,
                    batch=1,
                    augment=False,
                    mix_up=False,
                    epochs=1, with_embeddings=True):

        if episode_num is None:
            if phase == 'train':
                episode_num = 20000
            else:
                episode_num = 600

        num_samples_per_class = shot_num + episode_test_sample_num
        dataset = None
        for _ in tqdm(range(epochs)):
            if with_embeddings:
                data_set, global_label_depth, name_projector, embedding_datasets = self.generate_data_list(way_num,
                                                                                                           num_samples_per_class,
                                                                                                           episode_num=episode_num,
                                                                                                           phase=phase,
                                                                                                           with_embeddings=with_embeddings)
            else:
                data_set, global_label_depth, name_projector = self.generate_data_list(way_num,
                                                                                       num_samples_per_class,
                                                                                       episode_num=episode_num,
                                                                                       phase=phase,
                                                                                       with_embeddings=with_embeddings)

            ds = tf.data.Dataset.from_tensor_slices(data_set)

            if augment is True:
                process = process_with_augment
            else:
                process = process_without_augment

            ds_episode = ds.map(partial(process,
                                        way_num=way_num,
                                        episode_test_sample_num=episode_test_sample_num,
                                        global_label_depth=global_label_depth),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch, drop_remainder=True)

            if with_embeddings:
                e_ds, e_block, e_cluster = embedding_datasets
                e_ds = tf.data.Dataset.from_tensor_slices(e_ds)
                e_block = tf.data.Dataset.from_tensor_slices(e_block)
                e_cluster = tf.data.Dataset.from_tensor_slices(e_cluster)
                ds_episode_embedding = e_ds.map(partial(process_embedding,
                                                        way_num=way_num,
                                                        episode_test_sample_num=episode_test_sample_num),
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch,
                                                                                                        drop_remainder=True)
                ds_episode_embedding_blocks = e_block.map(partial(process_embedding_blocks,
                                                                  way_num=way_num, padding=1500),
                                                          num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch,
                                                                                                                  drop_remainder=True)
                ds_episode_embedding_cluster = e_cluster.map(partial(process_embedding_blocks,
                                                                     way_num=way_num, padding=10),
                                                             num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
                    batch,
                    drop_remainder=True)
                ds_episode = tf.data.Dataset.zip(
                    (ds_episode, ds_episode_embedding, ds_episode_embedding_blocks, ds_episode_embedding_cluster))

            if dataset is None:
                dataset = ds_episode
            else:
                dataset = dataset.concatenate(ds_episode)

        return dataset, name_projector

    def get_all_dataset(self, phase='train', batch=1, augment=False, contrastive=False):
        image_list, label_list, global_label_depth = self.generate_origin_data_list(phase)
        ds = tf.data.Dataset.from_tensor_slices((image_list, label_list)).shuffle(max(100000, len(image_list)))

        @tf.function
        def process(path, global_label, global_label_depth):
            image = read_image(path)
            global_label = tf.one_hot(global_label, axis=-1, depth=global_label_depth)
            return image, global_label

        if augment is True:
            process = process_with_crop
            if contrastive is True:
                process = process_with_crop_with_contrastive
        else:
            if contrastive is True:
                process = partial(process_with_crop_with_contrastive, p=0.)
        ds_episode = ds.map(partial(process, global_label_depth=global_label_depth),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch)
        return ds_episode, len(image_list)

    def get_embedding(self, model, redo=True, re_cluster=False, anchor_num=10, batch=64):

        from tqdm import tqdm
        from multiprocessing.dummy import Pool as ThreadPool
        from sklearn.cluster import KMeans
        try:
            kmeans = KMeans(n_clusters=anchor_num)
        except:
            kmeans = KMeans(n_clusters=anchor_num, n_init="auto")

        os.makedirs(self.embedding_dir_path, exist_ok=True)
        os.makedirs(os.path.join(self.embedding_dir_path, "train"), exist_ok=True);
        os.makedirs(os.path.join(self.embedding_dir_path, "test"), exist_ok=True);
        os.makedirs(os.path.join(self.embedding_dir_path, "val"), exist_ok=True);

        os.makedirs(self.embedding_block_dir_path, exist_ok=True)
        os.makedirs(os.path.join(self.embedding_block_dir_path, "train"), exist_ok=True);
        os.makedirs(os.path.join(self.embedding_block_dir_path, "test"), exist_ok=True);
        os.makedirs(os.path.join(self.embedding_block_dir_path, "val"), exist_ok=True);

        os.makedirs(self.embedding_clusters_dir_path, exist_ok=True)
        os.makedirs(os.path.join(self.embedding_clusters_dir_path, "train"), exist_ok=True);
        os.makedirs(os.path.join(self.embedding_clusters_dir_path, "test"), exist_ok=True);
        os.makedirs(os.path.join(self.embedding_clusters_dir_path, "val"), exist_ok=True);

        image_dict = self.meta_train_image_dict

        def process_save(data):
            z, path = data
            tf.io.write_file(path, tf.io.serialize_tensor(z))
            return z

        Z_block = []
        cluster_block = []
        for folderName, folderImageList in tqdm(image_dict.items()):
            embedding_dir = folderName.replace(self.base_dir, self.embedding_dir_path)
            embedding_path_list = [path.replace(self.base_dir, self.embedding_dir_path) + ".z" for path in
                                   folderImageList]
            cluster_dir = folderName.replace(self.base_dir, self.embedding_clusters_dir_path)
            block_dir = folderName.replace(self.base_dir, self.embedding_block_dir_path)
            os.makedirs(embedding_dir, exist_ok=True)
            os.makedirs(cluster_dir, exist_ok=True)
            os.makedirs(block_dir, exist_ok=True)
            block_path = os.path.join(block_dir, "block")
            cluster_path = os.path.join(cluster_dir, "cluster")
            Z = []
            if redo:
                all_train_ds = tf.data.Dataset.from_tensor_slices(folderImageList)
                all_train_ds = all_train_ds.map(partial(read_image),
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch)

                for x in all_train_ds:
                    Z.append(model(x, training=False))
                Z = tf.concat(Z, 0)
                process_save((Z, block_path))
                pool = ThreadPool(24)  # 创建10个容量的线程池并发执行
                pool.imap_unordered(process_save,
                                    zip(Z, embedding_path_list))

                pool.close()
                pool.join()
            else:
                if not os.path.exists(block_path):
                    print("!!!!!!!!!!!!Warnning ! {} is not exist! Now regenerate category embedding blocks....".format(
                        block_path))
                    all_train_ds = tf.data.Dataset.from_tensor_slices(embedding_path_list)
                    all_train_ds = all_train_ds.map(partial(load_embedding),
                                                    num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch)
                    for z_batch in tqdm(all_train_ds):
                        Z.append(z_batch)
                    Z = tf.concat(Z, 0)
                    process_save((Z, block_path))

                    re_cluster = True
                Z = tf.io.parse_tensor(tf.io.read_file(block_path), tf.float32)

            if redo or re_cluster:
                kmeans.fit(Z.numpy())
                anchors = kmeans.cluster_centers_
                process_save((anchors, cluster_path))
            else:
                anchors = tf.io.parse_tensor(tf.io.read_file(cluster_path), tf.float32)

        return
        #     Z = tf.pad(Z, [[0, 2048 - Z.shape[0]], [0, 0]])
        #     Z_block.append(Z)
        #     cluster_block.append(anchors)
        # print("done..!")
        # return tf.stack(Z_block, 0), tf.stack(cluster_block, 0),

    def get_embedding_block(self):
        os.makedirs(self.embedding_block_dir_path, exist_ok=True)
        os.makedirs(os.path.join(self.embedding_block_dir_path, "train"), exist_ok=True);
        os.makedirs(os.path.join(self.embedding_block_dir_path, "test"), exist_ok=True);
        os.makedirs(os.path.join(self.embedding_block_dir_path, "val"), exist_ok=True);

        os.makedirs(self.embedding_clusters_dir_path, exist_ok=True)
        os.makedirs(os.path.join(self.embedding_clusters_dir_path, "train"), exist_ok=True);
        os.makedirs(os.path.join(self.embedding_clusters_dir_path, "test"), exist_ok=True);
        os.makedirs(os.path.join(self.embedding_clusters_dir_path, "val"), exist_ok=True);


if __name__ == '__main__':
    data_dir_path = "/data/giraffe/0_FSL/data/mini_imagenet_tools/processed_images_origin"
    dataloader = MiniImageNetDataLoader(data_dir_path=data_dir_path)
    # meta_train_ds, meta_train_name_projector = dataloader.get_dataset_V2(phase='train', way_num=5, shot_num=5,
    #                                                                      episode_test_sample_num=5,
    #                                                                      episode_num=600,
    #                                                                      batch=4,
    #                                                                      augment=False)
    ds = dataloader.get_all_dataset(augment=True)

    for data in ds:
        img, y = data
        img = img * 255.
        cv2.imshow("image", img.numpy()[0].astype(np.uint8)[..., ::-1])
        cv2.waitKey(0)
        pass
