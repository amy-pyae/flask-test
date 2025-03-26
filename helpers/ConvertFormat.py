



def text_to_html(text):
    """Converts the given text to HTML format."""

    html = "<html><head><title>Dominate Singapore Search: A Comprehensive Guide to Local Keyword Research</title><meta name=\"description\" content=\"Keyword research is the magic key to unlocking online visibility in Singapore. This comprehensive guide unveils the secrets to potent keyword research, transforming your business's online presence. Discover how to unearth hidden gems, understand search intent, and conquer your niche. Learn strategies for effective keyword implementation and watch your website soar to the top of search results.\"></head><body>"

    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            level = line.count("#")
            html += f"<h{level}>{line.lstrip('# ').strip()}</h{level}>"
        elif line.startswith("Q:"):
            html += f"<p><strong>{line}</strong></p>"
        elif line.startswith("A:"):
            html += f"<p>{line}</p>"
        elif line:
            html += f"<p>{line}</p>"

    html += "</body></html>"
    return html