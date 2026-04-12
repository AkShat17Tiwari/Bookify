import ast
import astor
import sys

filename = 'app.py'
with open(filename, 'r') as f:
    source = f.read()

tree = ast.parse(source)

# We want to remove functions that have @app.route(...) and return render_template
# Actually, no, if we use astor, the formatting might change drastically.
# It's better to use a regex or string manipulation.
