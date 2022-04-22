gauge_obj = {
    'tooltip': {
        'formatter': '{a} <br/>{b} : {c}'
    },
    'series': [
        {
            'name': 'Gauge',
            'type': 'gauge',
            'startAngle': 180,
            'endAngle': 0,
            'min': 0,
            'max': 2500,
            'splitNumber': 5,
            'axisLine': {
                        'lineStyle': {
                            'width': 6,
                            'color': [
                                [0.20, '#FF6E76'],
                                [0.40, '#FDDD60'],
                                [0.60, '#7CFFB2'],
                                [1, '#58D9F9']
                            ]
                        }},
            'axisLabel': {
                'distance': 10,
                'fontSize': 8
            },
            'axisTick': {
                'length': 4,
                'lineStyle': {
                    'color': 'black',
                    'width': 1
                }
            },
            'splitLine': {
                'length': 8,
                'lineStyle': {
                    'color': 'black',
                    'width': 2
                }
            },
            'pointer': {
                'length': '60%',
                'width': 4,
                'itemStyle': {
                    'color': 'auto'
                }
            },
            'title': {
                'offsetCenter': [0, '0%'],
                'fontSize': 15,
                'color': 'grey'
            },
            'detail': {'fontSize': 20,
                       'formatter': '{value} kg/ha',
                       'color': 'auto',
                       'valueAnmiation': True,
                       'offsetCenter': [0, '+35%']},
            'data': [{'value': 0, 'name': None}]
        }
    ]
}