import copy
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

class DataLoader:
    def __init__(self, data_dir_path, multiView_embedding_path: list = None):
        self.base_dir = data_dir_path
        self.multi_view_dir_lists = multiView_embedding_path
        self.meta_test_folder = os.path.join(self.base_dir, "test")
        self.meta_val_folder = os.path.join(self.base_dir, "val")
        self.meta_train_folder = os.path.join(self.base_dir, "train")
        assert os.path.exists(self.meta_test_folder)
        assert os.path.exists(self.meta_val_folder)
        assert os.path.exists(self.meta_train_folder)

        self.meta_test_folder_lists = sorted([os.path.join(self.meta_test_folder, label) \
                                              for label in os.listdir(self.meta_test_folder) \
                                              if os.path.isdir(os.path.join(self.meta_test_folder, label)) \
                                              ])
        self.meta_test_global_label_dict = {foldername: index for index, foldername in
                                            enumerate(self.meta_test_folder_lists)}

        self.meta_test_image_dict = {foldername: os.listdir(foldername) for index, foldername in
                                     enumerate(self.meta_test_folder_lists)}
        self.meta_test_image_dict = {foldername: [] for index, foldername in
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
            image_dict = self.meta_train_image_dict
            label_dict = self.meta_train_global_label_dict
        elif phase == 'val':
            image_dict = self.meta_val_image_dict
            label_dict = self.meta_val_global_label_dict
        elif phase == 'test':
            image_dict = self.meta_test_image_dict
            label_dict = self.meta_test_global_label_dict
        else:
            print('Please select vaild phase')
            all_filenames = []

        print('Generating filenames')
        all_filenames = []
        class_id = []

        for k, v in image_dict.items():
            all_filenames.extend(v)
            class_id.extend([label_dict[k] for _ in range(len(v))])
        global_label_depth = len(label_dict)

        return all_filenames, class_id, global_label_depth

    def generate_data_list(self, way_num, num_samples_per_class, episode_num=None, phase='train'):
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

        for index in tqdm(range(episode_num)):
            sampled_character_folders = random.sample(folders, way_num)
            random.shuffle(sampled_character_folders)
            filenames = []
            labels = []
            global_labels = []
            for meta_label, cls in enumerate(sampled_character_folders):
                image_list = image_dict[cls]
                global_label = label_dict[cls]
                images = random.sample(image_list, num_samples_per_class)
                filenames.extend([os.path.join(cls, name) for name in images])
                labels.extend([meta_label for _ in images])
                global_labels.extend([global_label for _ in images])

            all_filenames[index] = filenames
            all_meta_labels[index] = labels
            all_global_labels[index] = global_labels
        return (all_filenames, all_meta_labels, all_global_labels), global_label_depth, name_projector

    def generate_data_list_no_putback(self, way_num, num_samples_per_class, episode_num=None, phase='train'):
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
        temp_image_dict = copy.deepcopy(image_dict)

        flag_of_sampled = set()
        for image_list in image_dict.values():
            flag_of_sampled = flag_of_sampled.union(set(image_list))
        total_count = len(flag_of_sampled)

        bar = tqdm(range(episode_num))
        for index in bar:
            bar.set_description("Processing %s" % "per:{} %".format(100. - 100 * len(flag_of_sampled) / total_count))
            sampled_character_folders = random.sample(folders, way_num)
            random.shuffle(sampled_character_folders)
            filenames = []
            labels = []
            global_labels = []
            for meta_label, cls in enumerate(sampled_character_folders):
                image_list = temp_image_dict[cls]
                global_label = label_dict[cls]
                if len(image_list) > num_samples_per_class:
                    images = random.sample(image_list, num_samples_per_class)
                    temp_image_dict[cls] = list(set(temp_image_dict[cls]) - set(images))
                    flag_of_sampled = flag_of_sampled - set(images)
                else:
                    images = copy.deepcopy(image_list)
                    rest_images = random.sample(image_dict[cls], num_samples_per_class - len(images))
                    temp_image_dict[cls] = list(set(image_dict[cls]) - set(rest_images))
                    images = images + rest_images
                    flag_of_sampled = flag_of_sampled - set(images)
                filenames.extend([os.path.join(cls, name) for name in images])
                labels.extend([meta_label for _ in images])
                global_labels.extend([global_label for _ in images])

            all_filenames[index] = filenames
            all_meta_labels[index] = labels
            all_global_labels[index] = global_labels

        return (all_filenames, all_meta_labels, all_global_labels), global_label_depth, name_projector

    def get_dataset(self, phase='train', way_num=5, shot_num=5, episode_test_sample_num=15, episode_num=None,
                    batch=1,
                    epochs=1,
                    cache_file="_cache_file"
                    ):

        if episode_num is None:
            if phase == 'train':
                episode_num = 20000
            else:
                episode_num = 600

        num_samples_per_class = shot_num + episode_test_sample_num
        dataset = None

        def loadNpEembedding(z_path):
            z = tf.io.parse_tensor(tf.io.read_file(z_path), tf.float32)
            return z

        @tf.function
        def split_shuffle_multi(x, y, way_num, shot_num, episode_test_sample_num):
            x_all = tf.stack(x, axis=0)
            x = tf.stack(tf.split(x_all, way_num, axis=1), axis=1)
            y, g_y = y
            y = tf.stack(tf.split(y, way_num), axis=0)
            g_y = tf.stack(tf.split(g_y, way_num), axis=0)

            indices = tf.random.shuffle(tf.range(episode_test_sample_num + shot_num))
            support_indice = indices[:shot_num]
            query_indice = indices[shot_num:]
            support_x = tf.gather(x, support_indice, axis=2)
            surport_y = tf.gather(y, support_indice, axis=1)
            surport_g_y = tf.gather(g_y, support_indice, axis=1)

            query_x = tf.gather(x, query_indice, axis=2)
            query_y = tf.gather(y, query_indice, axis=1)
            query_g_y = tf.gather(g_y, query_indice, axis=1)

            return (support_x, surport_y, surport_g_y), (query_x, query_y, query_g_y)

        @tf.function
        def split_shuffle(x, y, way_num, shot_num, episode_test_sample_num):
            x = tf.stack(tf.split(x, way_num), axis=0)
            y, g_y = y
            y = tf.stack(tf.split(y, way_num), axis=0)
            g_y = tf.stack(tf.split(g_y, way_num), axis=0)

            indices = tf.random.shuffle(tf.range(episode_test_sample_num + shot_num))
            support_indice = indices[:shot_num]
            query_indice = indices[shot_num:]
            support_x = tf.gather(x, support_indice, axis=1)
            surport_y = tf.gather(y, support_indice, axis=1)
            surport_g_y = tf.gather(g_y, support_indice, axis=1)

            query_x = tf.gather(x, query_indice, axis=1)
            query_y = tf.gather(y, query_indice, axis=1)
            query_g_y = tf.gather(g_y, query_indice, axis=1)

            return (support_x, surport_y, surport_g_y), (query_x, query_y, query_g_y)

        @tf.function
        def process(path, meta_label, global_label, way_num, shot_num, episode_test_sample_num,
                    global_label_depth):
            # z = tf.vectorized_map(
            #     lambda x: tf.io.parse_tensor(tf.io.read_file(x),tf.float32),path)
            x = tf.map_fn(loadNpEembedding, path, dtype=tf.float32, parallel_iterations=16)
            x.set_shape([path.shape[0], e_dim])
            meta_y = tf.one_hot(meta_label, axis=-1, depth=way_num)
            y = tf.one_hot(global_label, axis=-1, depth=global_label_depth)

            #
            # alfa_shape = tf.unstack(tf.shape(path)[:1])
            # alfa = tf.random.uniform(alfa_shape, 0.3, 1.0)
            # alfa_x = tf.reshape(alfa, [*alfa_shape, 1])
            #
            # indices = tf.random.shuffle(tf.range(tf.shape(x)[0]))
            # x_reference = tf.gather(x, indices)
            # y_reference = tf.gather(y, indices)
            # meta_y_reference = tf.gather(meta_y, indices)
            # # 保持 相同类别,和不同类别的混合
            # x = x * alfa_x + x_reference * (1.0 - alfa_x)
            # alfa_y = tf.reshape(alfa, [*alfa_shape, 1])
            # y = y * alfa_y + y_reference * (1.0 - alfa_y)
            # meta_y = meta_y * alfa_y + meta_y_reference * (1.0 - alfa_y)
            @tf.function
            def split(x, y, g_y, way_num, shot_num, episode_test_sample_num):
                x = tf.stack(tf.split(x, way_num), axis=0)
                y = tf.stack(tf.split(y, way_num), axis=0)
                g_y = tf.stack(tf.split(g_y, way_num), axis=0)

                # begin = tf.random.shuffle(tf.range(episode_test_sample_num ))[0]
                # end = begin + shot_num
                # # tf.print(begin, end)
                # support_x = x[:, begin:end, ...]
                # surport_y = y[:, begin:end:, ...]
                # surport_g_y = g_y[:, begin:end:, ...]
                #
                # query_x = tf.concat([x[:, 0:begin, ...], x[:, end:, ...]], 1)
                # query_y = tf.concat([y[:, 0:begin, ...], y[:, end:, ...]], 1)
                # query_g_y = tf.concat([g_y[:, 0:begin, ...], g_y[:, end:, ...]], 1)

                indices = tf.random.shuffle(tf.range(episode_test_sample_num + shot_num))
                support_indice = indices[:shot_num]
                # tf.print(x.shape,support_indice)
                query_indice = indices[shot_num:]
                support_x = tf.gather(x, support_indice, axis=1)
                surport_y = tf.gather(y, support_indice, axis=1)
                surport_g_y = tf.gather(g_y, support_indice, axis=1)

                query_x = tf.gather(x, query_indice, axis=1)
                query_y = tf.gather(y, query_indice, axis=1)
                query_g_y = tf.gather(g_y, query_indice, axis=1)

                # query_x = x[:, :episode_test_sample_num, ...]
                # query_y = y[:, :episode_test_sample_num, ...]
                # query_g_y = g_y[:, :episode_test_sample_num, ...]
                #
                # support_x = x[:, episode_test_sample_num:, ...]
                # surport_y = y[:, episode_test_sample_num:, ...]
                # surport_g_y = g_y[:, episode_test_sample_num:, ...]

                return (support_x, surport_y, surport_g_y), (query_x, query_y, query_g_y)

            return split(x, meta_y, y, way_num, shot_num, episode_test_sample_num)

        @tf.function
        def process_without_split(path, meta_label, global_label, way_num,
                                  global_label_depth):
            x = process_read_data(path)
            meta_y, y = process_read_label(meta_label, global_label, way_num,
                                           global_label_depth)
            return x, (meta_y, y)

        @tf.function
        def process_without_split2(path, meta_label, global_label, way_num,
                                   global_label_depth):
            path, path_r = path[0], path[1]
            x1 = process_read_data(path)
            x2 = process_read_data(path_r)
            meta_y, y = process_read_label(meta_label, global_label, way_num,
                                           global_label_depth)
            return (x1, x2), (meta_y, y)

        @tf.function
        def process_read_data(path):
            x = tf.map_fn(loadNpEembedding, path, dtype=tf.float32, parallel_iterations=32)
            x = tf.pad(x, [[0, 0], [0, e_dim - tf.shape(x)[-1]]])
            return x

        @tf.function
        def process_read_label(meta_label, global_label, way_num,
                               global_label_depth):
            meta_y = tf.one_hot(meta_label, axis=-1, depth=way_num)
            y = tf.one_hot(global_label, axis=-1, depth=global_label_depth)
            return meta_y, y

        for _ in tqdm(range(epochs)):

            data_set, global_label_depth, name_projector = self.generate_data_list(way_num,
                                                                                   num_samples_per_class,
                                                                                   episode_num=episode_num,
                                                                                   phase=phase)
            if self.multi_view_dir_lists is not None and len(self.multi_view_dir_lists) > 0:
                data_set_Refe = [[[fname.replace(self.base_dir, vd) for fname in e] for e in data_set[0]] for vd in
                                 self.multi_view_dir_lists]
                dsx = tf.data.Dataset.from_tensor_slices(data_set[:1])
                dsy = tf.data.Dataset.from_tensor_slices(data_set[1:])
                ds_x = dsx.map(partial(process_read_data),
                               num_parallel_calls=tf.data.experimental.AUTOTUNE)
                ds_y = dsy.map(partial(process_read_label,
                                       way_num=way_num,
                                       global_label_depth=global_label_depth),
                               num_parallel_calls=tf.data.experimental.AUTOTUNE)
                dsGroups = [ds_x]
                for ref in data_set_Refe:
                    dsGroups.append(tf.data.Dataset.from_tensor_slices(ref)
                                    .map(partial(process_read_data),
                                         num_parallel_calls=tf.data.experimental.AUTOTUNE))
                ds_episode = tf.data.Dataset.zip((tuple(dsGroups), ds_y))
                # dsGroups = [data_set[0], data_set_Refe[0]]
                # ds = tf.data.Dataset.from_tensor_slices((list(zip(*dsGroups)), *list(data_set[1:])))
                # ds_episode = ds.map(partial(process_without_split2,
                #                             way_num=way_num,
                #                             global_label_depth=global_label_depth),
                #                     num_parallel_calls=tf.data.experimental.AUTOTUNE)

            else:
                ds = tf.data.Dataset.from_tensor_slices(data_set)
                ds_episode = ds.map(partial(process_without_split,
                                            way_num=way_num,
                                            global_label_depth=global_label_depth),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

            if dataset is None:
                dataset = ds_episode
            else:
                dataset = dataset.concatenate(ds_episode)
            #
            # x,y = list(dataset.take(1))[0]
            # partial(split_shuffle,
            #         way_num=way_num,
            #         shot_num=shot_num,
            #         episode_test_sample_num=episode_test_sample_num)(x,y )
            # print("cache_file {}".format(cache_file))
            # # if os.path.exists(cache_file):
            # #     print("#################!!!!!!!~~~~~~~~~~~~~cache_file {} is exist, remove it: ".format(cache_file))
            # os.remove(cache_file)
            dataset = dataset.cache()
            if self.multi_view_dir_lists is not None and len(self.multi_view_dir_lists) > 0:
                dataset = dataset.map(partial(split_shuffle_multi,
                                              way_num=way_num,
                                              shot_num=shot_num,
                                              episode_test_sample_num=episode_test_sample_num),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch,
                                                                                              drop_remainder=True)
            else:
                dataset = dataset.map(partial(split_shuffle,
                                              way_num=way_num,
                                              shot_num=shot_num,
                                              episode_test_sample_num=episode_test_sample_num),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch,
                                                                                              drop_remainder=True)
        return dataset, name_projector

    def generate_anchors(self, phase="train", anchor_num=16, show=False):
        from sklearn.cluster import KMeans
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        image_list, label_list, global_label_depth = self.generate_origin_data_list(phase)
        ds = tf.data.Dataset.from_tensor_slices((image_list, label_list))
        batch = 256

        def loadNpEembedding(z_path):
            z = tf.io.parse_tensor(tf.io.read_file(z_path), tf.float32)
            return z

        def process(z_path, global_label, global_label_depth):
            z = loadNpEembedding(z_path)
            global_label = tf.one_hot(global_label, axis=-1, depth=global_label_depth)
            return z, global_label

        ds_episode = ds.map(partial(process, global_label_depth=global_label_depth),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch)
        embedding_dict = {}
        for data in tqdm(ds_episode):
            x, y = data
            for z, label in zip(x, tf.argmax(y, -1).numpy()):
                if label not in embedding_dict.keys():
                    embedding_dict[label] = tf.expand_dims(z, 0)
                else:
                    embedding_dict[label] = tf.concat([embedding_dict[label], tf.expand_dims(z, 0)], 0)

        def estimate(input):
            k, z = input
            z = z.numpy()
            try:
                kmeans = KMeans(n_clusters=anchor_num)
            except:
                kmeans = KMeans(n_clusters=anchor_num, n_init="auto")

            kmeans.fit(z)
            anchors = kmeans.cluster_centers_
            # out = {"predictor": kmeans, "anchors": anchors, "all_embeddings": z}
            out = {"anchors": anchors, "all_embeddings": z}
            clusters_dict = {}
            for sub_z, index in zip(z, kmeans.labels_):
                if index not in clusters_dict.keys():
                    clusters_dict[index] = tf.expand_dims(sub_z, 0)
                else:
                    clusters_dict[index] = tf.concat([clusters_dict[index], tf.expand_dims(sub_z, 0)], 0)
            out.update({"clusters": clusters_dict})

            if show:
                labels = kmeans.labels_
                centers = kmeans.cluster_centers_
                # Plot the data points with different colors for each cluster
                tsne = TSNE(n_components=2, n_jobs=8)
                X_tsne = tsne.fit_transform(np.concatenate([z, centers], 0))

                plt.scatter(X_tsne[:-anchor_num, 0], X_tsne[:-anchor_num, 1], c=labels)
                # Plot the cluster centers as black dots
                plt.scatter(X_tsne[-anchor_num:, 0], X_tsne[-anchor_num:, 1], c='black', s=200, alpha=0.5)
                plt.show()
            return k, out

        from multiprocessing.dummy import Pool as ThreadPool
        if show:
            pool = ThreadPool(1)
        else:
            pool = ThreadPool(4)  # 创建10个容量的线程池并发执行
        if anchor_num == 1:
            out = [estimate(data) for data in tqdm( embedding_dict.items())]
        else:
            out = list(tqdm(pool.imap_unordered(estimate, embedding_dict.items()), total=len(embedding_dict.items())))
        pool.close()
        pool.join()
        embedding_achor_dict = {k: outdict for k, outdict in out}
        anchor_z = []
        anchor_y = []
        for k, v in embedding_achor_dict.items():
            anchor_z.extend(v["anchors"])
            anchor_y.extend([k for _ in range(len(v["anchors"]))])
        anchor_z = tf.stack(anchor_z, 0)
        if show:
            tsne = TSNE(n_components=2, n_jobs=8)
            X_tsne = tsne.fit_transform(anchor_z)
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=anchor_y)
            colors = list(mcolors.XKCD_COLORS.keys())
            for i in range(X_tsne.shape[0]):
                text = anchor_y[i]
                color = colors[-text]
                plt.text(X_tsne[i, 0], X_tsne[i, 1], text,
                         color=color,
                         fontdict={'weight': 'bold', 'size': 15})
                plt.scatter(X_tsne[i, 0], X_tsne[i, 1], marker='o', color=color, edgecolors=color, s=20)
            plt.show()
        return embedding_achor_dict, tf.cast(anchor_z, tf.float32), tf.cast(tf.one_hot(anchor_y, global_label_depth),
                                                                            tf.float32)

    def get_all_dataset(self, phase='train', batch=1, shuffle=True):
        image_list, label_list, global_label_depth = self.generate_origin_data_list(phase)
        ds = tf.data.Dataset.from_tensor_slices((image_list, label_list))
        if shuffle:
            ds.shuffle(max(100000, len(image_list)))

        def loadNpEembedding(z_path):
            z = tf.io.parse_tensor(tf.io.read_file(z_path), tf.float32)
            return z

        def process(z_path, global_label, global_label_depth):
            z = loadNpEembedding(z_path)
            global_label = tf.one_hot(global_label, axis=-1, depth=global_label_depth)
            return z, global_label

        ds_episode = ds.map(partial(process, global_label_depth=global_label_depth),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch)
        return ds_episode, len(image_list)


if __name__ == '__main__':
    data_dir_path = "/data/giraffe/0_FSL/SE_FS_embedding"
    dataloader = DataLoader(data_dir_path=data_dir_path)
    # out = dataloader.generate_anchors(show=False,anchor_num=1)
    meta_train_ds, meta_train_name_projector = dataloader.get_dataset(phase='train', way_num=5, shot_num=5,
                                                                      episode_test_sample_num=15,
                                                                      episode_num=600,
                                                                      batch=4)

    for index, data in tqdm(meta_train_ds):
        print(" ")
    # print(tf.argmax(data[0][2], -1), tf.argmax(data[1][2], -1))
    # meta_train_ds, steps_per_epoch = dataloader.get_all_dataset(phase='train', batch=256 )
    # z_list, label_list, global_label_depth = dataloader.generate_origin_data_list('train')
    # ds = tf.data.Dataset.from_tensor_slices((z_list, label_list))
    #
    # embeddings = []
    # lables = []
    # for x, y in tqdm(meta_train_ds):
    #     embeddings.extend(x)
    #     lables.extend(y)
    # show_TSNE(embeddings,lables)
