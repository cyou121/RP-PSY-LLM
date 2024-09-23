import os
import json
import argparse
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from config import BIG5


def score_report(answers):
    dimension_wise_scores = {dim: [] for dim in BIG5}
    for _, ans in answers.items():
        dim = ans['dimension']

        # convert scores properly.
        # score = extract_and_convert(ans['answer'])
        score = ans['choice']

        if score in [1, 2, 3, 4, 5]:
            # reverse the scores according to the math sign.
            if dim != 'NEU' and ans['math'] == '-':
                score = 5 - (score - 1)
            elif dim == 'NEU' and ans['math'] == '+':
                score = 5 - (score - 1)

        dimension_wise_scores[dim].append(score)

    # get average score of all dimensions.
    final_score = {}
    for dim in BIG5:
        scores_dim = dimension_wise_scores[dim]
        scores_dim_f = [v for v in scores_dim if v is not None]

        if scores_dim_f:
            avg_util = sum(scores_dim_f) / len(scores_dim_f)
            final_score[dim] = avg_util
        else:
            final_score[dim] = None

    return final_score


def visualize_word_cloud(dim_wise_text, n_group=6):
    for dim in BIG5:
        dim_text = dim_wise_text[dim]
        sorted_text = [text for sc, text in sorted(
            dim_text, key=lambda x: x[0])]

        # split into n groups in ascending order.
        group_size = len(sorted_text) // n_group + \
            (len(sorted_text) % n_group)
        groups = [sorted_text[i:i+group_size]
                  for i in range(0, len(sorted_text), group_size)]

        # for each group generate a word cloud.
        fig, axs = plt.subplots(2, 3, figsize=(12, 6))
        axs = axs.flatten()
        group_repr = ['---', '--', '-', '+', '++', '+++']
        for i, group_texts in enumerate(groups):
            wordcloud = WordCloud(repeat=True).generate(
                ' '.join(group_texts))
            axs[i].imshow(wordcloud)
            axs[i].set_title(f"{dim}{group_repr[i]}")

        plt.tight_layout()
        plt.show()


def corr_analysis(dim_wise_scores):
    for dim in BIG5:
        dim_data = dim_wise_scores[dim]
        x, y = zip(*dim_data)

        # Calculate the correlation coefficient
        corr, p_value = pearsonr(x, y)
        print(f"[{dim}] Pearson Correlation coefficient:",
              round(corr, 3), p_value < 0.05)


def visualize_scatter_plot(dim_wise_scores, title=None, x_label=None, y_label=None):
    # place 5 subplots.
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
    ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2)
    ax4 = plt.subplot2grid((2, 6), (1, 1), colspan=2)
    ax5 = plt.subplot2grid((2, 6), (1, 3), colspan=2)

    # set scatter plots.
    for dim, ax in zip(BIG5, [ax1, ax2, ax3, ax4, ax5]):
        dim_data = dim_wise_scores[dim]
        x, y = zip(*dim_data)
        ax.scatter(x, y)

        ax.set_xlabel(x_label or f'{dim} ctrl vector strength')
        ax.set_ylabel(y_label or 'IPIP average score')

    # show the plot.
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def personality_based_analysis(all_data):
    dim_wise_scores = {dim: [] for dim in BIG5}
    for d in all_data:
        personality_dim, scalar = d['agent']
        scores = score_report(d['answers'])
        score = scores[personality_dim]

        # extract pairs of (dimension-scalar, ipip average score).
        if score is not None:
            dim_wise_scores[personality_dim].append((scalar, score))

    # Pearson correlation test.
    corr_analysis(dim_wise_scores)

    # visualize, correlation test.
    visualize_scatter_plot(dim_wise_scores,  # title='IPIP test analysis',
                           x_label=None, y_label='IPIP average score')


def generated_text_analysis(all_data, n_group=6, scalar_abs=True):
    dim_wise_text = {dim: [] for dim in BIG5}
    dim_wise_perplexity = {dim: [] for dim in BIG5}
    for d in all_data:
        personality_dim, scalar = d['agent']

        # extract self-intro.
        dim_wise_text[personality_dim].append((scalar, d['self_intro']))

        # extract perplexity.
        if scalar_abs:
            scalar = abs(scalar)
        dim_wise_perplexity[personality_dim].append((scalar, d["perplexity"]))

    # show the results.
    if scalar_abs:
        corr_analysis(dim_wise_perplexity)  # Pearson correlation test.
    visualize_scatter_plot(dim_wise_perplexity,  # visualize, correlation test.
                           x_label=None, y_label='Perplexity')

    visualize_word_cloud(dim_wise_text, n_group=n_group)  # word-clouds.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--output_dir', default='./tmp')
    parser.add_argument('--ipip_analysis',
                        default=False, action='store_true')
    parser.add_argument('--generation_analysis',
                        default=False, action='store_true')
    args = parser.parse_args()

    # load files from the directory.
    all_data = []
    for root, dirs, files in os.walk(args.output_dir):
        for fn in files:
            if not fn.endswith('json'):
                continue
            with open(os.path.join(root, fn), 'r') as F:
                data = json.load(F)
            all_data += data
    print(f'Extracted {len(all_data)} data points from {args.output_dir}')

    # analyze the generated texts, perplexity, IPIP scores.
    if args.ipip_analysis:
        personality_based_analysis(all_data)

    if args.generation_analysis:
        generated_text_analysis(all_data, n_group=6, scalar_abs=False)
