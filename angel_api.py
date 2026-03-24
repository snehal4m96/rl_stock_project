from datetime import datetime, timedelta

def get_historical_data(client, token):
    try:
        to_date = datetime.now()
        from_date = to_date - timedelta(days=365*5)

        params = {
            "exchange": "NSE",
            "symboltoken": token,
            "interval": "ONE_DAY",
            "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
            "todate": to_date.strftime("%Y-%m-%d %H:%M")
        }

        data = client.getCandleData(params)

        if data is None or 'data' not in data or len(data['data']) == 0:
            return None

        return data['data']

    except Exception as e:
        print("Error:", e)
        return None