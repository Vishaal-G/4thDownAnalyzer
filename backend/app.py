from flask import Flask, request, jsonify
from decision_engine import recommend, recommend_wp, feature_names

app = Flask(__name__)

@app.route('/api/recommend', methods=['POST'])
def view():
    # silent=True makes get_json() return None instead of raising on bad/missing JSON,
    # so we can turn that into a clean 400 response instead of an unhandled 500.
    inputData = request.get_json(silent=True)

    if inputData is None:
        return jsonify({'error': 'Request body must be valid JSON'}), 400

    missingFields = [name for name in feature_names if name not in inputData]
    if missingFields:
        return jsonify({'error': f'Missing required field(s): {", ".join(missingFields)}'}), 400

    try:
        result = recommend(**inputData)
        # WP nested under its own key so the EPA response shape (GO/PUNT/
        # FIELD_GOAL/BEST at the top level) stays unchanged for any existing
        # caller -- this is purely additive.
        result['WP'] = recommend_wp(**inputData)
    except TypeError as e:
        # Covers unexpected extra fields (unexpected keyword argument) and
        # wrong-typed values (e.g. a string where a number is expected, which
        # fails inside recommend()'s/recommend_wp()'s arithmetic).
        return jsonify({'error': f'Invalid input: {e}'}), 400

    return result

if __name__ == '__main__':
    app.run(debug=True)
