"""FileSystem MCP Server——基于沙箱的文件系统操作服务。

通过 MCP 协议暴露安全的文件操作工具，兼容标准 MCP 客户端。
限制操作在指定的根目录下，防止目录遍历攻击。

核心工具：
- read_file: 读取文件内容
- write_file: 写入文件内容
- list_directory: 列出目录内容
- move_file: 移动或重命名文件

启动方式（stdio）：
    python -m chainmaster.mcp.server.filesystem
"""

import logging
import os
import shutil
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# 配置日志记录
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_MCP_FS_ROOT_ENV = "MCP_FS_ROOT"
# 默认沙箱目录在当前工作目录的 data/sandbox/
_DEFAULT_FS_ROOT = "data/sandbox/"

mcp = FastMCP(
    "ChainMaster FileSystem",
    instructions=(
        "你是一个安全的文件系统管理器。你只能在预配置的沙箱目录下操作文件。"
        "提供了读取、写入、列出目录和移动文件等基本功能。"
    ),
)


def _get_sandbox_root() -> Path:
    """获取沙箱根目录，并确保其存在。"""
    root_str = os.environ.get(_MCP_FS_ROOT_ENV, _DEFAULT_FS_ROOT)
    root_path = Path(root_str).resolve()
    
    if not root_path.exists():
        logger.info(f"沙箱根目录 {root_path} 不存在，正在创建...")
        root_path.mkdir(parents=True, exist_ok=True)
        
    return root_path


def _validate_path(path_str: str) -> Path:
    """验证路径是否在沙箱根目录下，防止目录遍历攻击。"""
    root_path = _get_sandbox_root()
    target_path = Path(path_str)
    
    if target_path.is_absolute():
        resolved_path = target_path.resolve()
    else:
        resolved_path = (root_path / target_path).resolve()
        
    try:
        resolved_path.relative_to(root_path)
    except ValueError:
        error_msg = f"安全检查失败：路径 '{path_str}' 越界访问沙箱外部目录。"
        logger.error(error_msg)
        raise PermissionError(error_msg)
        
    return resolved_path


@mcp.tool()
def read_file(path: str) -> str:
    """读取指定路径的文件内容。"""
    logger.info(f"读取文件: {path}")
    try:
        target_path = _validate_path(path)
        if not target_path.exists():
            return f"错误：文件 '{path}' 不存在。"
        if not target_path.is_file():
            return f"错误：'{path}' 不是一个文件。"
            
        return target_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        return f"读取文件失败: {str(e)}"


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """向指定路径写入内容。如果文件已存在将被覆盖，如果父目录不存在将被自动创建。"""
    logger.info(f"写入文件: {path}")
    try:
        target_path = _validate_path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")
        return f"成功写入文件: {path}"
    except Exception as e:
        logger.error(f"写入文件失败: {e}")
        return f"写入文件失败: {str(e)}"


@mcp.tool()
def list_directory(path: str = ".") -> str:
    """列出指定目录下的所有文件和子目录。默认列出沙箱根目录。"""
    logger.info(f"列出目录: {path}")
    try:
        target_path = _validate_path(path)
        if not target_path.exists():
            return f"错误：目录 '{path}' 不存在。"
        if not target_path.is_dir():
            return f"错误：'{path}' 不是一个目录。"
            
        items = []
        for item in target_path.iterdir():
            item_type = "DIR " if item.is_dir() else "FILE"
            items.append(f"[{item_type}] {item.name}")
            
        if not items:
            return "目录为空。"
        return "\n".join(sorted(items))
    except Exception as e:
        logger.error(f"列出目录失败: {e}")
        return f"列出目录失败: {str(e)}"


@mcp.tool()
def move_file(source: str, destination: str) -> str:
    """移动或重命名文件/目录。source 和 destination 都必须在沙箱内。"""
    logger.info(f"移动文件: {source} -> {destination}")
    try:
        src_path = _validate_path(source)
        dst_path = _validate_path(destination)
        
        if not src_path.exists():
            return f"错误：源路径 '{source}' 不存在。"
            
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dst_path))
        return f"成功移动: {source} -> {destination}"
    except Exception as e:
        logger.error(f"移动文件失败: {e}")
        return f"移动文件失败: {str(e)}"


def main() -> None:
    """启动 FileSystem MCP Server（stdio 传输）。"""
    logger.info("正在启动 FileSystem MCP Server...")
    mcp.run()


if __name__ == "__main__":
    main()
