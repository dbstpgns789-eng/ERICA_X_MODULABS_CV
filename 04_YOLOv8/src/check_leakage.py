"""영상 프레임 데이터셋의 train/valid 누수 확인.

파일명에서 프레임 번호를 뽑아, valid 프레임마다 가장 가까운 train 프레임까지의
거리를 계산한다. 거리가 1~2면 사실상 같은 장면이 양쪽에 있다는 뜻이다.
"""
import bisect
import os
import re
from collections import Counter


def frame_numbers(split_dir):
    nums = []
    for f in os.listdir(os.path.join(split_dir, "images")):
        m = re.match(r"clip\d+_mp4-(\d+)_", f)
        if m:
            nums.append(int(m.group(1)))
    return sorted(nums)


def nearest_distances(query, reference):
    out = []
    for n in query:
        i = bisect.bisect_left(reference, n)
        cands = [abs(reference[j] - n) for j in (i - 1, i) if 0 <= j < len(reference)]
        out.append(min(cands))
    return out


def report(dataset_root):
    train = frame_numbers(os.path.join(dataset_root, "train"))
    valid = frame_numbers(os.path.join(dataset_root, "valid"))
    dists = nearest_distances(valid, train)
    c = Counter(dists)
    total = len(dists)
    print(f"valid {total}장 중 train 프레임과의 거리:")
    for d in sorted(c)[:5]:
        print(f"  거리 {d}: {c[d]}장 ({100 * c[d] / total:.0f}%)")
    within2 = sum(v for k, v in c.items() if k <= 2)
    print(f"  거리 2 이내: {within2}/{total} ({100 * within2 / total:.0f}%)")


if __name__ == "__main__":
    import sys
    report(sys.argv[1] if len(sys.argv) > 1 else "datasets/stamp")
