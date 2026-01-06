# 12306_cal

把 12306 通知邮件导出的 txt 解析为结构化 JSON，并生成一个可离线打开的 HTML 报表（支持年/月筛选、车站/车次/价格/席别/状态筛选）。

## 使用方法

### 1) 把 txt 放到 `data/`

例如：`data/hcy_12306.txt`、`data/xtc_12306.txt`。未来新增 txt 直接丢进 `data/` 也可以复现。

### 2) 运行解析器

在项目目录执行：

```bash
python3 scripts/parse_12306.py --input-dir data --output out
```

也可以显式指定文件：

```bash
python3 scripts/parse_12306.py --inputs data/hcy_12306.txt data/xtc_12306.txt --output out
```

### 3) 查看输出

解析后会生成：

- `out/events.jsonl`
- `out/tickets.json`
- `out/metadata.json`
- `out/report.html`（双击打开即可）

另外会按来源文件单独导出一份（方便分别统计/备份）：\n+\n+- `out/by_file/hcy_12306/events.jsonl`\n+- `out/by_file/hcy_12306/tickets.json`\n+- `out/by_file/xtc_12306/events.jsonl`\n+- `out/by_file/xtc_12306/tickets.json`\n+\n+报表支持按 **乘车人（passengerName）** 下拉筛选，筛选后图表/Top/明细会同步更新。\n+
> 如果不想生成报表：加 `--no-report`。


# 感谢

- **车票票**APP
- 邮箱自动提取参考：https://thedafeige.feishu.cn/docx/XW2td4j1xoGU1hxTs9vct0Crn3b