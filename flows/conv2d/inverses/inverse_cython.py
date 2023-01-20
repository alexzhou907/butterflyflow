import numpy as np
import time
# from Cython.Build import cythonize
import pyximport
# pyximport.install(
#     inplace=True,
#     )
pyximport.install(setup_args={
                              "include_dirs":np.get_include()},
                  reload_support=True,
inplace=True,)
import  .conv2d.inverses.inverse_op_cython as inverse_op_cython


class Inverse():
    ''' this only implements same channel number'''
    def __init__(self, is_upper, dilation):
        self.is_upper = is_upper
        self.dilation = dilation

    def __call__(self, z, w, b, channel_last=True):
        if np.isnan(z).any():
            return z

        # start = time.time()

        z = z - b

        if not channel_last:
            # change to channel last
            z = np.transpose(z, (0,2,3,1))
            w = np.transpose(w, (2,3,1,0))

        z_np = np.array(z, dtype='float64')
        w_np = np.array(w, dtype='float64')

        ksize = w_np.shape[0]
        kcent = (ksize - 1) // 2

        diagonal = np.diag(w_np[kcent, kcent, :, :])

        alpha = 1. / np.min(np.abs(diagonal))
        alpha = max(1., alpha)

        w_np *= alpha

        x_np = inverse_op_cython.inverse_conv(
            z_np, w_np, int(self.is_upper), self.dilation)

        x_np *= alpha

        if not channel_last:
            x_np = np.transpose(x_np, (0,3,1,2))
        # print('Inverse \t alpha {} \t compute time: {:.2f} seconds'.format(
        #                                         alpha, time.time() - start))
        return x_np.astype('float32')


class Inverse2():
    def __init__(self, is_upper, dilation):
        self.is_upper = is_upper
        self.dilation = dilation

    def __call__(self, z, w, b, channel_last=True):
        if np.isnan(z).any():
            return z

        print('Inverse called')
        start = time.time()

        # Subtract bias term.
        z = z - b


        if not channel_last:
            # change to channel last
            z = np.transpose(z, (0,2,3,1))
            w = np.transpose(w, (2,3,0,1))

        zs = z.shape
        batchsize, height, width, n_channels = zs
        ksize = w.shape[0]
        kcenter = (ksize-1) // 2


        diagonal = np.diag(w[kcenter, kcenter, :, :])
        # print(diagonal[np.argsort(diagonal)])
        factor = 1./np.min(diagonal)
        factor = max(1, factor)
        factor = 1.

        print('factor is', factor)
        # print('w is', w.transpose(3, 2, 0, 1))

        x_np = np.zeros(zs)
        z_np = np.array(z, dtype='float64')
        w_np = np.array(w, dtype='float64')

        w_np *= factor

        def filter2image(j, i, m, k):
            m_ = (m - kcenter) * self.dilation
            k_ = (k - kcenter) * self.dilation
            return j+k_, i+m_

        def in_bound(idx, lower, upper):
            return (idx >= lower) and (idx < upper)

        def reverse_range(n, reverse):
            if reverse:
                return range(n)
            else:
                return reversed(range(n))

        for b in range(batchsize):
            for j in reverse_range(height, self.is_upper):
                for i in reverse_range(width, self.is_upper):
                    for c_out in reverse_range(n_channels, not self.is_upper):
                        for c_in in range(n_channels):
                            for k in range(ksize):
                                for m in range(ksize):
                                    if k == kcenter and m == kcenter and \
                                            c_in == c_out:
                                        continue

                                    j_, i_ = filter2image(j, i, m, k)

                                    if not in_bound(j_, 0, height):
                                        continue

                                    if not in_bound(i_, 0, width):
                                        continue

                                    x_np[b, j, i, c_out] -= \
                                        w_np[k, m, c_in, c_out] \
                                        * x_np[b, j_, i_, c_in]

                        # Compute value for x
                        x_np[b, j, i, c_out] += z_np[b, j, i, c_out]
                        x_np[b, j, i, c_out] /= \
                            w_np[kcenter, kcenter, c_out, c_out]

        x_np = x_np * factor

        if not channel_last:
            x_np = np.transpose(x_np, (0,3,1,2))
        print('Total time to compute inverse {:.2f} seconds'.format(
                                                        time.time() - start))
        return x_np.astype('float32')
if __name__ == '__main__':
    inv = Inverse(1, 1)
    n,c,h,w = 2, 100, 5,5
    ksize = 3
    x = np.random.randn(n,c,h,w)
    w = np.ones([c,c,ksize,ksize])
    b = np.ones([1,c,1,1])

    out=inv(x,w,b,False)

    print()