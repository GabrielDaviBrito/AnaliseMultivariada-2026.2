import matplotlib.pyplot as plt

def biplot(scores, loadings, labels, var_names, pc=(0, 1), ax=None):
    """Recebe os scores, loadings, labels e nomes das variáveis e plota o biplot correspondente.

    Args:
        scores (_type_): Recebe os scores do PCA.
        loadings (_type_): Recebe os loadings do PCA.
        labels (_type_): Recebe os rótulos das observações.
        var_names (_type_): Recebe os nomes das variáveis.
        pc (tuple, optional): Tupla com os índices dos componentes principais a serem plotados. Defaults to (0, 1).
        ax (_type_, optional): Eixo do matplotlib. Defaults to None.
    """
    i, j = pc
    ax = ax or plt.gca()
    x, y = scores[:, i], scores[:, j]
 
    sx = 1.0 / (x.max() - x.min())
    sy = 1.0 / (y.max() - y.min())
    ax.scatter(x * sx, y * sy, alpha=0.7, color='steelblue')
    for k, txt in enumerate(labels):
        ax.annotate(txt, (x[k] * sx, y[k] * sy), fontsize=7, alpha=0.8)
 
    # setas das variáveis (cargas)
    for k, nome in enumerate(var_names):
        ax.arrow(0, 0, loadings[k, i], loadings[k, j],
                 color='crimson', head_width=0.02, length_includes_head=True)
        ax.text(loadings[k, i] * 1.12, loadings[k, j] * 1.12, nome,
                color='crimson', fontsize=11, fontweight='bold',
                ha='center', va='center')
 
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(0, color='gray', lw=0.5, ls='--')
    ax.set_xlabel(f'CP{i+1}')
    ax.set_ylabel(f'CP{j+1}')
    ax.set_title('Biplot – PCA (matriz de correlação)')
    ax.grid(alpha=0.2)
