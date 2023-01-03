import h5py
import torch.nn.functional as F
import os
from sklearn.preprocessing import QuantileTransformer
from torch.nn.utils.rnn import pad_sequence
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.ndimage import rotate
import matplotlib as mpl
from matplotlib.colors import LogNorm, LinearSegmentedColormap
mpl.use("Agg")


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > ./tmp')
    memory_available = [int(x.split()[2])
                        for x in open('tmp', 'r').readlines()]
    if len(memory_available) > 0:
        id = int(np.argmax(memory_available))
        print("setting to gpu:%d" % id)
        torch.cuda.set_device(id)
        return "cuda:%d" % id
    else:
        return


if torch.cuda.is_available():
    current_device = get_free_gpu()
else:
    current_device = 'cpu'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def proba2matrix(sample, weight=None, proba=None, intra=True):

    sample_left = sample
    weight_left = weight
    if intra:
        sample_left -= np.min(sample_left)
        size = int(np.max(sample_left) + 1)
        m = np.zeros((size, size), dtype='float32')
        if weight is not None:
            for i in range(sample_left.shape[-1] - 1):
                for j in range(i + 1, sample_left.shape[-1]):
                    m[sample_left[:, i], sample_left[:, j]
                      ] += np.maximum(proba * weight_left, proba)

        else:
            for i in range(sample_left.shape[-1] - 1):
                for j in range(i + 1, sample_left.shape[-1]):
                    m[sample_left[:, i], sample_left[:, j]] += proba

        m = m + m.T
    else:
        size1 = int(np.max(sample_left[:, 0]) - np.min(sample_left[:, 0]) + 1)
        size2 = int(np.max(sample_left[:, 1]) - np.min(sample_left[:, 1]) + 1)
        m = np.zeros((size1, size2), dtype='float32')
        if weight is not None:
            m[sample_left[:, 0] - np.min(sample_left[:, 0]), sample_left[:, 1] - np.min(
                sample_left[:, 1])] += np.maximum(proba * weight_left, proba)

        else:
            m[sample_left[:, 0] - np.min(sample_left[:, 0]),
              sample_left[:, 1] - np.min(sample_left[:, 1])] += proba

    return m


def generate_pair_wise(chrom_id):
    samples = []
    for i in range(chrom_range[chrom_id, 0], chrom_range[chrom_id, 1]):
        for j in range(i + min_dis, chrom_range[chrom_id, 1]):
            samples.append([i, j])

    samples = np.array(samples)
    return samples


def predict(model, input):
    model.eval()
    output = []
    new_batch_size = int(1e4)
    with torch.no_grad():
        for j in trange(math.ceil(len(input) / new_batch_size)):
            x = input[j *
                      new_batch_size:min((j + 1) * new_batch_size, len(input))]
            x = np2tensor_hyper(x, dtype=torch.long)
            x = pad_sequence(x, batch_first=True, padding_value=0).to(device)
            output.append(model(x).detach().cpu().numpy())
    output = np.concatenate(output, axis=0)
    torch.cuda.empty_cache()
    return output


config = get_config()
min_dis = config['min_distance']
temp_dir = config['temp_dir']
res = config['resolution']

chrom_range = np.load(os.path.join(temp_dir, "chrom_range.npy"))
classifier_model = torch.load(os.path.join(
    temp_dir, "model2load"), map_location=current_device)

print("device", classifier_model.layer_norm1.weight.device)
device_info = classifier_model.layer_norm1.weight.device
device_info = str(device_info).split(":")[-1]
torch.cuda.set_device(int(device_info))
transformer = QuantileTransformer(
    n_quantiles=1000, output_distribution='uniform')
task_mode = 'class'

origin = np.load(os.path.join(temp_dir, "intra_adj.npy")).astype('float32')
# origin = np.load(os.path.join(temp_dir, "edge_list_adj.npy")).astype('float32')
origin_hic = np.load('/scratch/2022-11-02/bio-xujs/HiPore-C_Promoter_Result/Result/HyperGraph/GM12878_HiPore-C_hypergraph/GSE63525_GM12878_HiC/intra_adj.npy').astype('float32')

# generating mcool file
f = h5py.File( os.path.join(temp_dir, 'plot_denoise/denoised.mcool'), 'w')
# f = h5py.File("../denoised.mcool", "w")
grp = f.create_group("resolutions")
grp = grp.create_group("%d" % res)
cooler_bin = grp.create_group("bins")
node2bin = np.load(os.path.join(temp_dir, "node2bin.npy"),
                   allow_pickle=True).item()

chrom_list = []
chrom_start = []
chrom_end = []
chrom_name = config['chrom_list']

for i in range(1, int(np.max(list(node2bin.keys())) + 1)):
    bin_ = node2bin[i]
    chrom, start = bin_.split(":")
    chrom_list.append(chrom_name.index(chrom))
    chrom_start.append(int(start))
    chrom_end.append(int(start) + res)

print(np.array(chrom_list), np.array(chrom_start))
cooler_bin.create_dataset("chrom", data=chrom_list)
cooler_bin.create_dataset("start", data=chrom_start)
cooler_bin.create_dataset("end", data=chrom_end)

cooler_chrom = grp.create_group("chroms")
cooler_chrom.create_dataset("name", data=[l.encode(
    'utf8') for l in chrom_name], dtype=h5py.special_dtype(vlen=str))

cooler_pixels = grp.create_group("pixels")
bin_id1 = []
bin_id2 = []
balanced = []


for i in range(len(chrom_name)):
    pair_wise = generate_pair_wise(i)
    print(pair_wise)
    bin_id1.append(np.copy(pair_wise[:, 0]) - 1)
    bin_id2.append(np.copy(pair_wise[:, 1]) - 1)
    # print (pair_wise.shape, pair_wise)
    proba = predict(classifier_model, pair_wise).reshape((-1))
    if task_mode == 'class':
        proba = torch.sigmoid(torch.from_numpy(proba)).numpy()
    else:
        proba = F.softplus(torch.from_numpy(proba)).numpy()
    # print ( np.sum(proba >= 0.5) ,proba.shape)

    pair_wise_weight = np.array([origin[e[0] - 1, e[1] - 1]
                                for e in tqdm(pair_wise)])

    my_proba = proba2matrix(pair_wise, None, proba)
    coverage1 = np.sqrt(np.mean(my_proba, axis=-1, keepdims=True))
    coverage2 = np.sqrt(np.mean(my_proba, axis=0, keepdims=True))
    my_proba = my_proba / (coverage1 + 1e-15)
    my_proba = my_proba / (coverage2 + 1e-15)

    pair_wise_hic_weight = np.array([origin_hic[e[0] - 1, e[1] - 1]
                                for e in tqdm(pair_wise)])
    origin_hic_part = proba2matrix(pair_wise, None, pair_wise_hic_weight)
    gap1 = np.sum(origin_hic_part, axis=-1) == 0
    gap2 = np.sum(origin_hic_part, axis=0) == 0
    coverage1 = np.sqrt(np.mean(origin_hic_part, axis=-1, keepdims=True))
    coverage2 = np.sqrt(np.mean(origin_hic_part, axis=0, keepdims=True))
    origin_hic_part = origin_hic_part / (coverage1 + 1e-15)
    origin_hic_part = origin_hic_part / (coverage2 + 1e-15)

    origin_part = proba2matrix(pair_wise, None, pair_wise_weight)
    gap1 = np.sum(origin_part, axis=-1) == 0
    gap2 = np.sum(origin_part, axis=0) == 0
    coverage1 = np.sqrt(np.mean(origin_part, axis=-1, keepdims=True))
    coverage2 = np.sqrt(np.mean(origin_part, axis=0, keepdims=True))
    origin_part = origin_part / (coverage1 + 1e-15)
    origin_part = origin_part / (coverage2 + 1e-15)

    my = my_proba * origin_part
    my = np.maximum(my_proba * origin_part, my_proba)
    coverage1 = np.sqrt(np.mean(my, axis=-1, keepdims=True))
    coverage2 = np.sqrt(np.mean(my, axis=0, keepdims=True))
    my = my / (coverage1 + 1e-15)
    my = my / (coverage2 + 1e-15)

    my[gap1, :] = 0.0
    my[:, gap2] = 0.0
    my_proba[gap1, :] = 0.0
    my_proba[:, gap2] = 0.0

    my = transformer.fit_transform(my.reshape((-1, 1))).reshape((len(my), -1))
    origin_hic_part = transformer.fit_transform(
        origin_hic_part.reshape((-1, 1))).reshape((len(origin_hic_part), -1))
    origin_part = transformer.fit_transform(
        origin_part.reshape((-1, 1))).reshape((len(origin_part), -1))
    my_proba = transformer.fit_transform(
        my_proba.reshape((-1, 1))).reshape((len(my), -1))

    np.save(os.path.join(temp_dir, f'plot_denoise/{chrom_name[i]}_denoised.npy'), my)
    np.save(os.path.join(temp_dir, f'plot_denoise/{chrom_name[i]}_origin.npy'), origin_part)
    np.save(os.path.join(temp_dir, f'plot_denoise/{chrom_name[i]}_origin_hic.npy'), origin_hic_part)

    with open(os.path.join(temp_dir, 'plot_denoise/Gm12878.1Mb.denoise.pairs'), 'a') as fw:
        my_triu = np.triu(my)
        for m in trange(len(my_triu)):
            for n in range(len(my_triu)):
                value = my_triu[m, n]
                if value != 0:
                    chrom = chrom_name[i]
                    tmp_ = chrom_range[i][0] - 1
                    node_m = tmp_ + m
                    node_n = tmp_ + n
                    chr_start_m = node2bin[node_m+1]
                    chr_m, start_m = chr_start_m.split(':')
                    chr_m = chr_m.replace('chr', '')
                    chr_start_n = node2bin[node_n+1]
                    chr_n, start_n = chr_start_n.split(':')
                    chr_n = chr_n.replace('chr', '')
                    fw.write(f'{chr_m}\t{start_m}\t{chr_n}\t{start_n}\t{value}\n')

    with open(os.path.join(temp_dir, 'plot_denoise/Gm12878.1Mb.origin.pairs'), 'a') as fw:
        origin_part_triu = np.triu(origin_part)
        for m in trange(len(origin_part_triu)):
            for n in range(len(origin_part_triu)):
                value = origin_part_triu[m, n]
                if value != 0:
                    chrom = chrom_name[i]
                    tmp_ = chrom_range[i][0] - 1
                    node_m = tmp_ + m
                    node_n = tmp_ + n
                    chr_start_m = node2bin[node_m+1]
                    chr_m, start_m = chr_start_m.split(':')
                    chr_m = chr_m.replace('chr', '')
                    chr_start_n = node2bin[node_n+1]
                    chr_n, start_n = chr_start_n.split(':')
                    chr_n = chr_n.replace('chr', '')
                    fw.write(f'{chr_m}\t{start_m}\t{chr_n}\t{start_n}\t{value}\n')

    with open(os.path.join(temp_dir, 'plot_denoise/Gm12878.1Mb.origin_hic.pairs'), 'a') as fw:
        origin_hic_part_triu = np.triu(origin_hic_part)
        for m in trange(len(origin_hic_part_triu)):
            for n in range(len(origin_hic_part_triu)):
                value = origin_hic_part_triu[m, n]
                if value != 0:
                    chrom = chrom_name[i]
                    tmp_ = chrom_range[i][0] - 1
                    node_m = tmp_ + m
                    node_n = tmp_ + n
                    chr_start_m = node2bin[node_m+1]
                    chr_m, start_m = chr_start_m.split(':')
                    chr_m = chr_m.replace('chr', '')
                    chr_start_n = node2bin[node_n+1]
                    chr_n, start_n = chr_start_n.split(':')
                    chr_n = chr_n.replace('chr', '')
                    fw.write(f'{chr_m}\t{start_m}\t{chr_n}\t{start_n}\t{value}\n')

    my_triu = np.triu(my)
    rotate_my_triu = rotate(my_triu, angle=45)
    length = rotate_my_triu.shape[0]
    vmin = 0
    vmax = np.percentile(rotate_my_triu, 99)
    cmap = LinearSegmentedColormap.from_list('normal', [(1, 1, 1), (1, 0, 0)], N=1000)
    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(hspace=0, wspace=0.02)
    mask = None
    # print ("matrix", matrix, np.min(matrix), np.max(matrix))
    # ax = sns.heatmap(rotate_my_triu, cmap=cmap, square=True, mask=mask,
    #                     cbar=False, vmin=vmin, vmax=vmax)
    left, bottom, width, height = 0.25, 0.5, 0.45, 0.45
    size_heatmap = [left, bottom, width, height/2]
    ax1 = fig.add_axes(size_heatmap)
    img = ax1.imshow(rotate_my_triu, cmap=cmap, origin='upper', interpolation='nearest',
                        extent=(0, length, 0, length),
                        aspect='auto', vmax=vmax, vmin=vmin)  # extent: (left, right, bottom, top)
    ax1.set_title(f'{chrom_name[i]}_denoise')
    ax1.set_ylim([length/2, length])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklines(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax1.get_yticklines(), visible=False)

    plt.savefig( os.path.join(temp_dir, f'plot_denoise/{chrom_name[i]}_denoise.pdf'), dpi=300)
    plt.close(fig)


    min_ = np.min(pair_wise)
    # print (pair_wise[:, 0] - min_, pair_wise[:, 1] - min_, my.shape)
    value = my[pair_wise[:, 0] - min_, pair_wise[:, 1] - min_]
    balanced.append(value)

    # fig = plt.figure(figsize=(5, 5))
    # plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    # mask =None
    # # print ("matrix", matrix, np.min(matrix), np.max(matrix))
    # ax = sns.heatmap(my_proba, cmap="Reds", square=True, mask=mask ,cbar=False, vmin=vmin, vmax=vmax)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # plt.savefig("../chr%d_denoise_proba.pdf" %(i+1), dpi=300)
    # plt.close(fig)


    origin_part_triu = np.triu(origin_part)
    rotate_origin_part_triu = rotate(origin_part_triu, angle=45)
    length =  rotate_origin_part_triu.shape[0]
    vmin = 0
    vmax = np.percentile( rotate_origin_part_triu, 99)
    cmap = LinearSegmentedColormap.from_list('normal', [(1, 1, 1), (1, 0, 0)], N=1000)
    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(hspace=0, wspace=0.02)
    mask = None
    left, bottom, width, height = 0.25, 0.5, 0.45, 0.45
    size_heatmap = [left, bottom, width, height/2]
    ax1 = fig.add_axes(size_heatmap)
    img = ax1.imshow( rotate_origin_part_triu, cmap=cmap, origin='upper', interpolation='nearest',
                        extent=(0, length, 0, length),
                        aspect='auto', vmax=vmax, vmin=vmin)  # extent: (left, right, bottom, top)
    ax1.set_title(f'{chrom_name[i]}_origin')
    ax1.set_ylim([length/2, length])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklines(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax1.get_yticklines(), visible=False)

    plt.savefig( os.path.join(temp_dir, f'plot_denoise/{chrom_name[i]}_origin.pdf'), dpi=300)
    plt.close(fig)


    origin_hic_part_triu = np.triu(origin_hic_part)
    rotate_origin_hic_part_triu = rotate(origin_hic_part_triu, angle=45)
    length =  rotate_origin_hic_part_triu.shape[0]
    vmin = 0
    vmax = np.percentile( rotate_origin_hic_part_triu, 99)
    cmap = LinearSegmentedColormap.from_list('normal', [(1, 1, 1), (1, 0, 0)], N=1000)
    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(hspace=0, wspace=0.02)
    mask = None
    left, bottom, width, height = 0.25, 0.5, 0.45, 0.45
    size_heatmap = [left, bottom, width, height/2]
    ax1 = fig.add_axes(size_heatmap)
    img = ax1.imshow( rotate_origin_hic_part_triu, cmap=cmap, origin='upper', interpolation='nearest',
                        extent=(0, length, 0, length),
                        aspect='auto', vmax=vmax, vmin=vmin)  # extent: (left, right, bottom, top)
    ax1.set_title(f'{chrom_name[i]}_origin_hic')
    ax1.set_ylim([length/2, length])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklines(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax1.get_yticklines(), visible=False)

    plt.savefig( os.path.join(temp_dir, f'plot_denoise/{chrom_name[i]}_origin_hic.pdf'), dpi=300)
    plt.close(fig)

bin_id1 = np.concatenate(bin_id1, axis=0)
bin_id2 = np.concatenate(bin_id2, axis=0)
balanced = np.concatenate(balanced, axis=0)
cooler_pixels.create_dataset("bin1_id", data=bin_id1)
cooler_pixels.create_dataset("bin2_id", data=bin_id2)
cooler_pixels.create_dataset("balanced", data=balanced)

