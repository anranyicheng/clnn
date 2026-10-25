#!/usr/bin/env bash
#
# run-tests.sh — 运行 clnn 测试套件
#
# 用法:
#   ./run-tests.sh                # 全部（含两个慢速套件）
#   ./run-tests.sh --fast         # 只跑快速套件（约 1 分钟）
#   ./run-tests.sh --slow         # 只跑慢速套件（约 5 分钟）
#   ./run-tests.sh test-architectures pdf-bugs-verify   # 指定若干文件（可省略 .lisp）
#
# 判定规则：进程退出码非 0 即视为失败（各测试文件都会在有失败项时
# 以非 0 退出；断言型的文件失败时会直接抛错）。
#
set -u

cd "$(dirname "$0")" || exit 1

export SBCL_HOME="${SBCL_HOME:-/usr/local/lib/sbcl}"
SBCL="${SBCL:-sbcl}"
HEAP="${CLNN_TEST_HEAP:-4096}"

# 快速套件：纯合成数据、无外部依赖，单文件 < 40 秒
FAST_TESTS=(
  clnn-test-suite
  test-architecture-fixes
  test-regression-fixes
  test-copy-network
  test-cache-slots
  test-conv-im2col
  test-cosine-loss-final
  test-all-fixes
  pdf-bugs-verify
  audit-verify
  test-architectures
  test-end-to-end-training
)

# 慢速套件：gradcheck 做全量有限差分，smoke 在真实 MNIST 上训练
SLOW_TESTS=(
  gradcheck-all-layers
  smoke-mnist-convergence
)

usage() {
  sed -n '3,12p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

SELECTED=()
case "${1:-}" in
  -h|--help) usage 0 ;;
  --fast)    SELECTED=("${FAST_TESTS[@]}") ;;
  --slow)    SELECTED=("${SLOW_TESTS[@]}") ;;
  "")        SELECTED=("${FAST_TESTS[@]}" "${SLOW_TESTS[@]}") ;;
  *)         for a in "$@"; do SELECTED+=("${a%.lisp}"); done ;;
esac

if [ ! -x "$(command -v "$SBCL")" ]; then
  echo "找不到 sbcl（可用 SBCL=/path/to/sbcl 指定）" >&2
  exit 127
fi

TMPDIR_TESTS="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_TESTS"' EXIT

PASSED=0
FAILED=0
FAILED_NAMES=()
START_ALL=$(date +%s)

echo "clnn 测试套件 —— 共 ${#SELECTED[@]} 个文件，堆 ${HEAP}MB"
echo "================================================================================"

for name in "${SELECTED[@]}"; do
  file="${name}.lisp"
  if [ ! -f "$file" ]; then
    echo "跳过 $file（文件不存在）"
    FAILED=$((FAILED + 1))
    FAILED_NAMES+=("$name(缺失)")
    continue
  fi

  log="$TMPDIR_TESTS/$name.log"
  start=$(date +%s)
  "$SBCL" --dynamic-space-size "$HEAP" --non-interactive \
          --eval '(require :asdf)' \
          --eval '(asdf:load-system :clnn)' \
          --load "$file" > "$log" 2>&1
  code=$?
  elapsed=$(( $(date +%s) - start ))

  # 摘一行结果摘要（各套件的汇总行格式不同，尽量抓最有信息量的一行）
  summary=$(grep -aE "结果: .*PASS|PASS, .*FAIL|passed, .*failed|通过: .*|个失败|Summary: |显式判定失败项" "$log" \
            | tail -1 | sed 's/^[[:space:]]*//')

  if [ "$code" -eq 0 ]; then
    PASSED=$((PASSED + 1))
    printf '  [PASS] %-28s %4ds  %s\n' "$name" "$elapsed" "$summary"
  else
    FAILED=$((FAILED + 1))
    FAILED_NAMES+=("$name")
    printf '  [FAIL] %-28s %4ds  %s\n' "$name" "$elapsed" "$summary"
  fi
done

TOTAL=$(( $(date +%s) - START_ALL ))
echo "================================================================================"
printf '总计: %d 个文件通过, %d 个失败, 耗时 %ds\n' "$PASSED" "$FAILED" "$TOTAL"

if [ "$FAILED" -gt 0 ]; then
  echo "失败文件: ${FAILED_NAMES[*]}"
  echo "（日志已保留在 $TMPDIR_TESTS；需要复查可单独运行对应 .lisp）"
  trap - EXIT          # 失败时保留日志
  exit 1
fi

exit 0
