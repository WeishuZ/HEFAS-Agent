import requests
import json
import os
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
import re
import time

class WebpageToRAGConverter:
    """将网页转换为RAG友好文档的类"""
    
    def __init__(self, url: str, doc_name: Optional[str] = None, api_key: str = None):
        """
        初始化转换器
        
        Args:
            url: 要抓取的网页URL
            doc_name: 可选的文档名称，如果不提供则从URL生成
            api_key: jina API密钥
        """
        self.url = url
        self.doc_name = doc_name if doc_name else self._generate_doc_name_from_url(url)
        self.content = None
        self.processed_content = None
        self.api_key = api_key
        
    def _generate_doc_name_from_url(self, url: str) -> str:
        """从URL生成文档名称，使用当前目录作为基础目录"""
        # 移除协议部分
        url = url.replace('http://', '').replace('https://', '')
        # 替换不适合作为文件名的字符
        url = re.sub(r'edu|www|html',"",url)
        url = url.replace('/', '\\').replace('.', '\\').replace(':', '\\').replace("\\\\",'\\')
        
        # 确保路径以相对路径开始，而不是绝对路径
        if url.startswith('\\'):
            url = url[1:]  # 移除开头的斜杠，使其成为相对路径
        if url.endswith('\\'):
            url = url[:-1]

            
            
        
        return f"data\\{url}"

    
    def fetch_content(self) -> Dict[str, Any]:
        """使用jina API抓取网页内容"""
        try:
            # 创建请求头
            headers = {
                "Authorization": f"Bearer {self.api_key}" if self.api_key else None
            }
            
            # 清理请求头，移除None值
            headers = {k: v for k, v in headers.items() if v is not None}
            
            # 构建jina URL
            jina_url = f'https://r.jina.ai/{self.url}'
            
            # 发送请求
            response = requests.get(jina_url, headers=headers)
            
            if response.status_code == 200:
                # 解析JSON响应
                try:
                    self.content = response.json()
                except json.JSONDecodeError:
                    # 如果不是JSON格式，则处理为文本
                    self.content = {
                        "title": "Extracted Content",
                        "content": response.text,
                        "url": self.url
                    }
                return self.content
            else:
                raise Exception(f"抓取失败，状态码: {response.status_code}，响应: {response.text}")
        except Exception as e:
            raise Exception(f"抓取网页内容时出错: {str(e)}")
    
    def process_for_rag(self) -> Dict[str, Any]:
        """获取jina原始响应数据用于RAG应用"""
        if not self.content:
            self.fetch_content()
        
        # 直接返回jina的原始响应数据
        # 不做额外处理或格式转换
        return self.content

    
    def save_to_file(self, format: str = "txt") -> str:
        """
        保存处理后的内容到文件
        
        Args:
            format: 输出格式，默认为txt
                
        Returns:
            保存的文件路径
        """
        # 确保文件名有正确的扩展名
        file_path = f"{self.doc_name}.{format.lower()}"
        dir_path = os.path.dirname(file_path)
        
        # 创建所有必要的目录
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if format.lower() == "json":
                    # 如果要求JSON格式，保存完整数据
                    json.dump(self.content, f, ensure_ascii=False, indent=2)
                elif format.lower() == "txt":
                    # 如果是txt格式，只提取content内容并添加源URL
                    text_content = ""
                    if isinstance(self.content, dict):
                        # 直接从content字段获取
                        if 'content' in self.content:
                            text_content = self.content['content']
                        # 嵌套在data中的情况
                        elif 'data' in self.content and isinstance(self.content['data'], dict):
                            if 'content' in self.content['data']:
                                text_content = self.content['data']['content']
                        # 找不到content字段
                        else:
                            text_content = str(self.content)
                    else:
                        text_content = str(self.content)
                    
                    # 添加源URL到文本开头
                    full_content = f"Source: {self.url}\n\n{text_content}"
                    f.write(full_content)
                else:
                    # 其他格式按原样保存
                    f.write(str(self.content))
            
            print(f"已保存内容到: {file_path}")
            return file_path
        except Exception as e:
            print(f"保存文件时出错: {e}")
            return ""

    
    def get_processed_content(self) -> Dict[str, Any]:
        """获取处理后的内容"""
        if not self.processed_content:
            self.process_for_rag()
        return self.processed_content


def process_url_list(urls: List[str], api_key: str = None, output_dir: Optional[str] = None, 
                     doc_names: Optional[List[str]] = None, 
                     max_workers: int = 5) -> List[Dict[str, Any]]:
    """
    将URL列表转换为RAG友好的文档
    
    Args:
        urls: 要处理的URL列表
        api_key: jina API密钥
        output_dir: 输出目录，默认为当前目录
        doc_names: 可选的文档名称列表，长度应与urls相同或为None
        max_workers: 并行处理的最大线程数
        
    Returns:
        处理后的文档列表
    """
    # 创建输出目录（如果不存在）
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    results = []
    failed_urls = []
    
    # 处理单个URL的函数
    def process_single_url(idx, url):
        try:
            doc_name = doc_names[idx] if doc_names and idx < len(doc_names) else None
            
            # 创建转换器实例
            converter = WebpageToRAGConverter(url, doc_name, api_key)
            
            # 处理内容
            processed_content = converter.process_for_rag()
            
            # 如果指定了输出目录，则保存到该目录
            if output_dir:
                file_name = converter.doc_name + ".txt"
                file_path = os.path.join(output_dir, file_name)
                converter.save_to_file(file_path)
                print(f"成功处理并保存: {url} -> {file_path}")
            else:
                saved_path = converter.save_to_file()
                print(f"成功处理: {url} -> {saved_path}")
            
            return processed_content
            
        except Exception as e:
            print(f"处理URL时出错 {url}: {str(e)}")
            failed_urls.append(url)
            return None
    
    # 使用线程池并行处理URL
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {executor.submit(process_single_url, i, url): i 
                          for i, url in enumerate(urls)}
        
        # 收集结果
        for future in future_to_idx:
            result = future.result()
            if result:
                results.append(result)
    
    # 报告处理结果
    print(f"\n处理总结:")
    print(f"成功处理的URL数量: {len(results)}")
    print(f"失败处理的URL数量: {len(failed_urls)}")
    
    if failed_urls:
        print("\n处理失败的URL:")
        for url in failed_urls:
            print(f"- {url}")
    
    return results
