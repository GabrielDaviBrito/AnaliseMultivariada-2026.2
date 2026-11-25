import numpy as np
import matplotlib.pyplot as plt


def biplot(scores, loadings, labels, var_names, pc=(0, 1), ax=None, label_fontsize=6):
    """Plota o biplot do PCA.

    Args:
        scores: Scores do PCA.
        loadings: Loadings do PCA.
        labels: Rótulos das observações.
        var_names: Nomes das variáveis.
        pc (tuple, optional): Índices dos componentes a plotar.
        ax (optional): Eixo do matplotlib.
        label_fontsize (int, optional): Tamanho da fonte dos rótulos das observações.
    """
    i, j = pc
    ax = ax or plt.gca()
    labels = np.asarray(labels)
    x, y = scores[:, i], scores[:, j]
    sx = 1.0 / (x.max() - x.min())
    sy = 1.0 / (y.max() - y.min())
    xs, ys = x * sx, y * sy

    ax.scatter(xs, ys, alpha=0.7, color='steelblue', s=15)

    textos = [
        ax.text(xs[k], ys[k], str(labels[k]), fontsize=label_fontsize, alpha=0.8)
        for k in range(len(labels))
    ]

    for k, nome in enumerate(var_names):
        ax.arrow(0, 0, loadings[k, i], loadings[k, j],
                 color='crimson', head_width=0.01, length_includes_head=True, alpha=0.7)
        ax.text(loadings[k, i] * 1.1, loadings[k, j] * 1.1, nome,
                color='crimson', fontsize=9, fontweight='bold',
                ha='center', va='center', alpha=0.8)

    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(0, color='gray', lw=0.5, ls='--')
    ax.set_xlabel(f'CP{i+1}')
    ax.set_ylabel(f'CP{j+1}')
    ax.set_title('Biplot – PCA (matriz de correlação)')
    ax.grid(alpha=0.2)
