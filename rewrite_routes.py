import re

with open('app.py', 'r') as f:
    content = f.read()

legacy_routes = [
    '/', '/auth', '/login', '/signup', '/logout', '/recommend', '/mood', 
    '/recommend_books', '/multimodal', '/profile', '/for_you', '/onboarding', 
    '/book/details', '/admin'
]

for route in legacy_routes:
    if route == '/':
        content = re.sub(r"@app\.route\('/'\)", r"@app.route('/legacy_index')", content)
        content = re.sub(r'@app\.route\("/"\)', r'@app.route("/legacy_index")', content)
        # Note: we need to handle trailing commas or methods string like @app.route('/', methods=...)
        content = re.sub(r"@app\.route\('/',", r"@app.route('/legacy_index',", content)
        content = re.sub(r'@app\.route\("/",', r'@app.route("/legacy_index",', content)
    else:
        escaped_route = route.replace('/', r'\/')
        rep = '/legacy_' + route.strip('/')
        content = content.replace(f"@app.route('{route}')", f"@app.route('{rep}')")
        content = content.replace(f'@app.route("{route}")', f'@app.route("{rep}")')
        content = content.replace(f"@app.route('{route}',", f"@app.route('{rep}',")
        content = content.replace(f'@app.route("{route}",', f'@app.route("{rep}",')

with open('app.py', 'w') as f:
    f.write(content)
