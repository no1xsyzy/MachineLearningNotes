机器学习笔记
====

编译需要[GNU make](https://www.gnu.org/software/make/)
、[pandoc](http://pandoc.org/)。

编译PDF还需要任意的latex引擎。在texlive下测试通过。

`*.md`除`README.md`以外均为pandoc markdown；
`header.tex`、`begin.tex`、`append.tex`为适应CJK字符集的latex代码

```bash
# pdf
make Exx.pdf
```
