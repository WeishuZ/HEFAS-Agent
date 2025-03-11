from typing import List, Dict, Any, Optional
import os
from concurrent.futures import ThreadPoolExecutor
import time
from web2Rag import WebpageToRAGConverter


def process_url_list(urls: List[str], output_dir: Optional[str] = None, 
                     doc_names: Optional[List[str]] = None, 
                     max_workers: int = 5) -> List[Dict[str, Any]]:
    """
    将URL列表转换为RAG友好的文档
    
    Args:
        urls: 要处理的URL列表
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
            converter = WebpageToRAGConverter(url, doc_name)
            
            # 处理内容
            processed_content = converter.process_for_rag()
            
            # 如果指定了输出目录，则保存到该目录
            if output_dir:
                file_name = converter.doc_name + ".txt"
                file_path = os.path.join(output_dir, file_name)
                converter.save_to_file(file_path)
                print(f"成功处理并保存: {url} -> {file_path}")
            else:
                converter.save_to_file()
                print(f"成功处理: {url}")
            
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


# 使用示例
if __name__ == "__main__":
    # 要处理的URL列表
    url_list = [
        "https://www.deanza.edu/hefas/",
        "https://www.deanza.edu/hefas/resources.html",
        "https://www.deanza.edu/hefas/ZoomHours.html",
        "https://www.deanza.edu/hefas/internbio.html",
        "https://www.deanza.edu/hefas/legislation.html"
        "https://www.deanza.edu/hefas/Members.html",
        "https://www.deanza.edu/hefas/undocustem.html",
        "https://www.deanza.edu/hefas/Internships.html",
        "https://www.deanza.edu/hefas/Volunteering.html",
        "https://www.deanza.edu/hefas/Summit.html",
        "https://www.deanza.edu/hefas/action-week.html",
        "https://www.deanza.edu/vida/undocuwelcome.html",
        "https://www.deanza.edu/hefas/undocusol.html",
        "https://www.deanza.edu/hefas/donations.hefas.html", 
    ]
    custom_names = []

    # 指定输出目录
    output_directory = ""
    
    # 处理URL列表
    start_time = time.time()
    processed_docs = process_url_list(
        urls=url_list,
        output_dir=output_directory,
        doc_names=custom_names,
        max_workers=3  # 同时处理3个URL
    )
    end_time = time.time()
    
    print(f"\n处理完成! 总耗时: {end_time - start_time:.2f}秒")
    print(f"文档已保存至目录: {os.path.abspath(output_directory)}")
    
    # 显示第一个文档的内容摘要
    if processed_docs:
        first_doc = processed_docs[0]
        print("\n第一个文档内容摘要:")
        print(f"标题: {first_doc.get('title')}")
        content = first_doc.get('content', '')
        print(f"内容预览: {content[:200]}..." if len(content) > 200 else content)
