# 下载与安装

从 [GitHub Releases](https://github.com/SCUliujiacheng/evidence-rag-bench-zh/releases)
选择 `v0.3.0` 预发布版。发布页附有下表中的四个应用包、`SHA256SUMS.txt`
校验文件和 `build-manifest.json` 构建清单。运行检查报告保存在对应 GitHub Actions
运行的 artifacts 中，不属于 Release 附件。自动生成的 “Source code” 压缩包
用于开发，直接使用请选择下表中的应用附件。

| 设备 | 应用附件 |
| --- | --- |
| Windows，64 位 Intel/AMD | `evidence-rag-bench-zh-0.3.0-windows-x64.zip` |
| Apple 芯片 Mac | `evidence-rag-bench-zh-0.3.0-macos-arm64.zip` |
| Intel 芯片 Mac | `evidence-rag-bench-zh-0.3.0-macos-x64.zip` |
| Android 7.0 及以上的 arm64 手机，或 x86_64 模拟器 | `evidence-rag-bench-zh-0.3.0-android.apk` |

Windows 请完整解压 ZIP，保留可执行文件旁的 `_internal` 文件夹，然后打开
`evidence-rag-bench-zh.exe`。Mac 解压后打开其中的 `.app`，也可以把它移入
“应用程序”。无需另装 Python。启动页会在默认浏览器中打开，提供“打开应用”
和“退出程序”按钮；关闭浏览器标签页不会停止本机进程。

首批附件是预览版：Windows 尚无发布者证书，macOS 尚未完成 Developer ID
签名和公证。系统可能显示未知发布者提示或阻止打开。请先核对仓库、版本和
`SHA256SUMS.txt`，再决定是否允许运行预览版本。自动运行检查不代表已通过真机认证。

Android 下载 APK 后，用系统安装器打开。系统可能要求允许当前下载应用安装
其他来源的应用。这是独立离线应用，不是应用商店页面；只支持 64 位设备，
不支持较早的 32 位 Android 手机。

两种版本都有中英文语料选择和原文证据查看界面，可以输入新问题，不需要
API Key 或网络。桌面版还提供评测 API；手机版使用便携词法检索引擎，不运行
完整评测 CLI 或 CrossEncoder。可选语义重排需要源码安装，常规桌面包和手机包
都没有附带该模型。具体边界见[版本说明](download-editions.md)。

Android 只允许应用内部页面导航，GitHub 和“原文”链接不会外跳到网站；结果中
仍可展开查看已保存的原始片段。桌面版联网时可以通过原文链接访问上游网站。

桌面评测报告与安装目录分开保存：

| 平台 | 报告目录 |
| --- | --- |
| Windows | `%LOCALAPPDATA%\evidence-rag-bench-zh\artifacts\reports` |
| macOS | `~/Library/Application Support/evidence-rag-bench-zh/artifacts/reports` |

手机版不保存问题或答案，运行文件位于应用私有存储。替换桌面应用文件夹不会
删除用户数据目录；删除用户目录或卸载 Android 应用前，请先另外保存需要保留
的报告或文件。应用不会向安装资源目录写入评测报告。
