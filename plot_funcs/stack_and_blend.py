import numpy as np

def stack_and_blend(arrA, arrB, overlap=200):
    """
    Stacks arrA on top of arrB with a smooth blend of 'overlap' rows in between.
    Both arrA and arrB must have the same number of columns.
    
    arrA.shape = (hA, w)
    arrB.shape = (hB, w)
    overlap is an integer number of rows to blend.
    
    Returns a single 2D array of shape (hA + hB - overlap, w).
    """
    hA, wA = arrA.shape
    hB, wB = arrB.shape
    assert wA == wB, "arrA and arrB must have the same width."
    if overlap < 1:
        overlap = 1
    if overlap > min(hA, hB):
        overlap = min(hA, hB)

    # Height of the final stacked array
    h_final = hA + hB - overlap
    blended = np.zeros((h_final, wA), dtype=arrA.dtype)

    # 1) Copy arrA rows except for the last 'overlap' part
    cutoffA = hA - overlap
    blended[:cutoffA, :] = arrA[:cutoffA, :]

    # 2) Blend region
    for i in range(overlap):
        # alpha goes from 0 -> 1 across the overlap
        alpha = i / float(overlap - 1)
        # row in the final array
        row_final = cutoffA + i
        # row in A and B
        rowA = cutoffA + i
        rowB = i
        # blend
        blended[row_final, :] = (1 - alpha) * arrA[rowA, :] + alpha * arrB[rowB, :]

    # 3) Copy the remaining rows of arrB
    blended[cutoffA + overlap:, :] = arrB[overlap:, :]

    return blended