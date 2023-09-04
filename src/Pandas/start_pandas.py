# Source - Real Python blog - 
# Official PANDAS - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html
# https://github.com/pydata/bottleneck

"""
Error - Check --- 
>>> pd.get_option('display.max_rows')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: module 'pandas' has no attribute 'get_option'
"""
def start():
    """
    # Don't wrap to multiple pages
    # Max length of printed sequence
    # Controls SettingWithCopyWarning
    """
    options = {
        'display': {
                'max_columns': None,
                'max_colwidth': 25,
                'expand_frame_repr': False,
                'max_rows': 14,
                'max_seq_items': 50,
                'precision': 4,
                'show_dimensions': False
                    },
        'mode': {
            'chained_assignment': None
                }
            }

    for category, option in options.items():
        for op, value in option.items():
            pd.set_option(f'{category}.{op}', value)

    if __name__ == '__main__':
        start()
        #del start # Clean up namespace in the interpreter