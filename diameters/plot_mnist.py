import parse_smt2
import numpy as np
import matplotlib.pyplot as plt
import sys

# fn = 'examples/mnist_itp.phi.txt'
fn = sys.argv[1]
fm = int(sys.argv[2])
to = int(sys.argv[3])

variables = [ 'x%d' % (n+1) for n in range(784) ]

def plot_single_ineq(ineq,ax):
    assert ineq.shape == (785,)

    nonconst_part = ineq[:-1].reshape((28,28))

#     plt.imshow(nonconst_part,interpolation=None)
#     plt.colorbar()
#     plt.title('Weighted sum must exceed %f' % (-ineq[-1]))
#     plt.show()
    img = ax.imshow(nonconst_part,interpolation=None)
    fig = ax.get_figure()
    fig.colorbar(img,ax=ax)
    ax.set_title('Weighted sum must exceed %f' % (-ineq[-1]))

def plot_multi_ineq(ineq):
    assert ineq.shape[1] == (785)
    plot_cnt = ineq.shape[0]
    rows = (plot_cnt // 10) + 1
    cols = 10
    fig,ax = plt.subplots(rows,cols,squeeze=False)
    ax = ax.reshape((-1,))

    nonconst_part = ineq[:,:-1].reshape((-1,28,28))
    const_part = ineq[:,-1]
    vmin = np.min(nonconst_part)
    vmax = np.max(nonconst_part)

    for pos in range(plot_cnt):
        img = ax[pos].imshow(nonconst_part[pos],interpolation=None,vmin=vmin,vmax=vmax)
        ax[pos].set_title('Σ >= %f' % (-ineq[pos,-1]))
    fig.colorbar(img,ax=ax,location='bottom')

def plot_all_ineq(ineq):
    F,X = plt.subplots(1,1)

    img = X.imshow(ineq,aspect='auto',interpolation='nearest')
    plt.colorbar(img,ax=X)


if __name__ == '__main__':
    data = open(fn,'r').read()
#    parsed = parse_smt2.parse_and_interpret(data,variables)[0]
#    np.save('/home/mwand/sandbox/mnistarr.npy',parsed)
    parsed = np.load('/home/mwand/sandbox/mnistarr.npy')

    # trest
#     F,X = plt.subplots(8,1)
# #     plot_single_ineq(parsed[1],X)
    plot_multi_ineq(parsed[fm:to])

#     plot_all_ineq(parsed[:10,:-1])
    plt.show()

