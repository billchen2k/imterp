import datetime
import os

import flask
from flask import abort, jsonify, send_from_directory
from natsort import natsorted

from utils import logger

app = flask.Flask(__name__)
result_dir = './results'

result_dir = os.path.abspath(result_dir)

@app.route('/')
def main():
    return jsonify({'msg': 'server running.'})


@app.route('/workdirs')
def workdirs():
    workdirs = os.listdir(os.path.join(result_dir))
    workdirs = natsorted(workdirs, reverse=False)
    return jsonify(list(map(lambda x: f'zterp/{x}', workdirs)))


@app.route('/res', defaults={'path': ''})
@app.route('/res/<path:path>', methods=['GET'])
def res(path):
    abs_path = os.path.join(result_dir, path)
    # logger.debug(abs_path)
    if not os.path.exists(abs_path) or not abs_path.startswith(os.path.abspath(result_dir)):
        return abort(404)

    if os.path.isdir(abs_path):
        items = os.listdir(abs_path)
        items = natsorted(items, reverse=True)
        html = f'''
        <h2>Files under directory {abs_path}:</h2>
        <ul>
        '''
        for item in items:
            mod_date = os.path.getmtime(os.path.join(abs_path, item))
            readable_mod = datetime.datetime.fromtimestamp(mod_date).strftime('%Y-%m-%d %H:%M:%S')
            item_path = os.path.join(path, item)
            html += f'''<li>
            <a href="/res/{item_path}">{item}</a> <span style="color: #888888; font-size: 0.8rem; user-select: none;">Mod: {readable_mod}</span>
            </li>'''
        html += '</ul>'
        return html

    # Serve file
    return send_from_directory(result_dir, path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5275, debug=True)
