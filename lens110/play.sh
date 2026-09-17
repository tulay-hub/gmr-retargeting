#!/usr/bin/env bash
# 便捷播放 lens110 动作, 支持 pkl / CSV 两种格式。
#
# 用法:
#   ./play.sh                          # 交互菜单: 选择格式 -> 选择文件 -> 速度
#   ./play.sh --pkl <文件.pkl> [--speed 0.8]   # 直接播放 pkl
#   ./play.sh --csv <文件.csv> [--speed 0.8]   # 直接播放 CSV
#
# 交互菜单里会自动列出全身项目 processed data 下的 pkl / csv 文件供选择。

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
GMR_PY="${GMR_PY:-python}"
MOTION_DIR="$REPO_ROOT/projects/01_dance_whole_body/data/processed/retargeted_actions"

if ! command -v "$GMR_PY" >/dev/null 2>&1; then
    echo "找不到 gmr 环境 python: $GMR_PY"
    echo "请先: conda activate gmr"
    exit 1
fi

cd "$REPO_ROOT"

play_pkl() { "$GMR_PY" "$HERE/play_lens110_pkl.py" --pkl "$@"; }
play_csv() { "$GMR_PY" "$HERE/play_lens110_csv.py" --csv "$@"; }

# ---------- 直接指定格式 ----------
if [ "$1" = "--pkl" ] || [ "$1" = "--csv" ]; then
    fmt="$1"
    shift
    if [ -z "$1" ]; then
        echo "缺少文件路径: ./play.sh $fmt <文件> [--speed 0.8]"
        exit 1
    fi
    if [ "$fmt" = "--pkl" ]; then
        play_pkl "$@"
    else
        play_csv "$@"
    fi
    exit 0
fi

# ---------- 交互菜单 ----------
echo "============== lens110 动作播放器 =============="
echo "选择播放格式:"
echo "  1) pkl"
echo "  2) csv"
printf "输入 1 或 2: "
read -r fmt_choice

case "$fmt_choice" in
    1) ext="pkl";   player=play_pkl ;;
    2) ext="csv";   player=play_csv ;;
    *) echo "无效选择: $fmt_choice"; exit 1 ;;
esac

mapfile -t files < <(find "$MOTION_DIR" -type f -name "*.$ext" | sort)
if [ "${#files[@]}" -eq 0 ]; then
    echo "在 $MOTION_DIR 下没有找到 .$ext 文件"
    exit 1
fi

echo ""
echo "可用的 .$ext 文件:"
for i in "${!files[@]}"; do
    printf "  %2d) %s\n" "$((i + 1))" "$(basename "${files[$i]}")"
done
printf "输入编号 (1-%d): " "${#files[@]}"
read -r file_choice

if ! [[ "$file_choice" =~ ^[0-9]+$ ]] || [ "$file_choice" -lt 1 ] || [ "$file_choice" -gt "${#files[@]}" ]; then
    echo "无效编号: $file_choice"
    exit 1
fi
file_path="${files[$((file_choice - 1))]}"

printf "播放速度 (回车默认 1.0, 0.5=半速): "
read -r speed
if [ -z "$speed" ]; then
    speed="1.0"
fi

echo ""
echo "播放: $file_path  (speed ${speed}x)"
"$player" "$file_path" --speed "$speed"
