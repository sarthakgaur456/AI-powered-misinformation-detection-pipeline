import urllib.robotparser

rp = urllib.robotparser.RobotFileParser()
rp.set_url("https://www.politifact.com/robots.txt")
rp.read()
path = "/factchecks/"
print("Can fetch index:", rp.can_fetch("*", "https://www.politifact.com" + path))
# Also check article pattern if you know it:
print("Can fetch article example:", rp.can_fetch("*", "https://www.politifact.com/factchecks/2025/aug/07/some-article/"))
