import mwclient
from urllib.parse import urlparse

from config import SMW_URL, SMW_USERNAME, SMW_PASSWORD


def process_smw_content(page_title):
    """从Semantic MediaWiki获取并处理内容"""
    # 连接到MediaWiki
    parsed = urlparse(SMW_URL)
    host = parsed.netloc or parsed.path
    path = parsed.path if parsed.path else "/"
    site = mwclient.Site(host, path=path)
    site.login(SMW_USERNAME, SMW_PASSWORD)

    # 获取页面内容
    page = site.pages[page_title]
    if not page.exists:
        raise Exception(f"页面 {page_title} 不存在")

    # 获取页面内容（可根据需要处理语义标记）
    content = page.text()

    # 简单处理：移除MediaWiki标记
    import re

    # 移除标题标记
    content = re.sub(r'=+([^=]+)=+', r'\1', content)

    # 移除链接标记
    content = re.sub(r'\[\[(.*?)\]\]', r'\1', content)

    # 移除模板标记
    content = re.sub(r'{{(.*?)}}', '', content)

    # 移除HTML标签
    content = re.sub(r'<.*?>', '', content)

    return content.strip()