from pathlib import Path
import os
import sys

"""确保项目的依赖和模块可以被正确地导入，并且提供了一种方便的方式来引用项目内部的文件和目录。"""

FILE = Path(__file__).resolve()#将当前文件的路径解析为一个Path对象，并用.resolve()转化为绝对路径
ROOT = FILE.parents[0]  # root directory
    #parents属性是Path对象的一个属性，它返回一个生成器，用于遍历当前路径的所有上级目录
    #parents[0]获取当前路径的父目录（即上级目录）的Path对象。这是一个相对路径的上一级目录。

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
    #如果项目的根目录不在sys.path中，这行代码将根目录添加到sys.path列表中。这样，Python解释器在导入模块时也会在项目的根目录中查找，确保项目内部的模块可以被正确导入。

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
#os.path.relpath函数返回一个相对路径，这个路径是从当前工作目录到ROOT的路径。