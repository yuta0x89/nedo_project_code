import numpy as np
import sys

def compare_arrays(file1, file2, epsilon):
    # NumPyファイルを読み込む
    hflogits = np.load(file1)
    mglogits = np.load(file2)

    print(f"hflogits {hflogits}")
    print(f"mglogits {mglogits}")

    # 一致する要素の数を計算
    same_num = (hflogits == mglogits).sum()

    # 差がepsilonを超える要素の数を計算
    diff_num = ((np.abs(hflogits - mglogits)) > epsilon).sum()

    # 最大差を計算
    diff_max = np.abs(hflogits - mglogits).max()

    # 誤差が5%以内の要素の数を計算
    within_5_percent = ((np.abs(hflogits - mglogits) / np.abs(hflogits)) <= 0.05).sum()
    within_10_percent = ((np.abs(hflogits - mglogits) / np.abs(hflogits)) <= 0.5).sum()

    # 結果を表示
    print(hflogits.shape)
    sums=hflogits.shape[0]*hflogits.shape[1]
    print(f'nuber of lements:{sums}' )
    print(f"Number of matching elements: {same_num}")
    print(f"Number of elements with difference greater than epsilon: {diff_num}")
    print(f"Maximum difference: {diff_max}")
    print(f"Number of elements within 5% error margin: {within_5_percent}")
    print(f"Number of elements within 50% error margin: {within_10_percent}")
    print("np.isclose(mglogits[:,:], hfflogits[:,:], rtol=1e-05, atol=1e-02).sum(axis=-1)")
    print(np.isclose(mglogits[:,:], hflogits[:,:], rtol=1e-05, atol=1e-02).sum(axis=-1))
    print("np.isclose(mglogits[:,:], hflogits[:,:], rtol=1e-04, atol=1e-02).sum(axis=-1)")
    print(np.isclose(mglogits[:,:], hflogits[:,:], rtol=1e-04, atol=1e-02).sum(axis=-1))
    print("np.isclose(mglogits[:,:], hflogits[:,:], rtol=1e-03, atol=1e-02).sum(axis=-1)")
    print(np.isclose(mglogits[:,:], hflogits[:,:], rtol=1e-03, atol=1e-02).sum(axis=-1))
    print("np.isclose(mglogits[:,:], hflogits[:,:], rtol=1e-02, atol=1e-02).sum(axis=-1)")
    print(np.isclose(mglogits[:,:], hflogits[:,:], rtol=1e-02, atol=1e-02).sum(axis=-1))
    print("np.isclose(mglogits[:,:], hflogits[:,:], rtol=1e-01, atol=1e-02).sum(axis=-1)")
    print(np.isclose(mglogits[:,:], hflogits[:,:], rtol=1e-01, atol=1e-02).sum(axis=-1))
    print()
    print("np.isclose(mglogits[:,:], hfflogits[:,:], rtol=1e-05, atol=1e-01).sum(axis=-1)")
    print(np.isclose(mglogits[:,:], hflogits[:,:], rtol=1e-05, atol=1e-01).sum(axis=-1))
    print("np.isclose(mglogits[:,:], hflogits[:,:], rtol=1e-04, atol=1e-01).sum(axis=-1)")
    print(np.isclose(mglogits[:,:], hflogits[:,:], rtol=1e-04, atol=1e-01).sum(axis=-1))
    print("np.isclose(mglogits[:,:], hflogits[:,:], rtol=1e-03, atol=1e-01).sum(axis=-1)")
    print(np.isclose(mglogits[:,:], hflogits[:,:], rtol=1e-03, atol=1e-01).sum(axis=-1))
    print("np.isclose(mglogits[:,:], hflogits[:,:], rtol=1e-02, atol=1e-01).sum(axis=-1)")
    print(np.isclose(mglogits[:,:], hflogits[:,:], rtol=1e-02, atol=1e-01).sum(axis=-1))
    print("np.isclose(mglogits[:,:], hflogits[:,:], rtol=1e-01, atol=1e-01).sum(axis=-1)")
    print(np.isclose(mglogits[:,:], hflogits[:,:], rtol=1e-01, atol=1e-01).sum(axis=-1))

    print(" ## starting checking softmax ##")
    h_index = np.argmax(hflogits, axis=1)
    m_index = np.argmax(mglogits, axis=1)
    print(h_index)
    print(m_index)

    acc = np.array([h_index==m_index]).sum() / len(h_index)    
    print(f"acc == {acc}")
    print(h_index.shape)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compare_arrays.py <file1.npy> <file2.npy> <epsilon>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    epsilon = float(sys.argv[3])

    compare_arrays(file1, file2, epsilon)
