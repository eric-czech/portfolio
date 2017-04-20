

def analyze_sets(s1, s2, names=None, stats=True, items=False):
    """
    Analyzes two sets of objects and returns intersection as well as differences
    :param s1: First set of objects; need not be a set -- it will be converted to one
    :param s2: Second set of objects; need not be a set -- it will be converted to one
    :param names: Optional 2-item sequence containing the names that should be used to refer to each set
        (if not specified, the sets will be referred to by their symbolic names (i.e. s1 & s2))
    :param stats: Boolean indicating whether or not to include statistics on set differences
    :param items: Boolean indicating whether or not to include actual items for set differences
    :return: Dict with summary statistics and/or actual items for set differences and intersections
    """
    s1 = set(s1)
    s2 = set(s2)
    n1 = names[0] if names else 'Set1'
    n2 = names[1] if names else 'Set2'

    in_both = s1 & s2
    in_one = s1 ^ s2
    in_s1 = s1 - s2
    in_s2 = s2 - s1
    union = s1 | s2
    n = len(union)

    res = {}
    if stats:
        res['Stats'] = {
            'InBoth': '{} ({:.2f}%)'.format(len(in_both), 100.*len(in_both)/n),
            'InOnlyOne': '{} ({:.2f}%)'.format(len(in_one), 100.*len(in_one)/n),
            'InOnly{}'.format(n1): '{} ({:.2f}%)'.format(len(in_s1), 100.*len(in_s1)/n),
            'InOnly{}'.format(n2): '{} ({:.2f}%)'.format(len(in_s2), 100.*len(in_s2)/n),
            'All': '{} (100%)'.format(n)
        }
    if items:
        res['Items'] = {
            'InBoth': in_both,
            'InOnlyOne': in_one,
            'InOnly{}'.format(n1): in_s1,
            'InOnly{}'.format(n2): in_s2,
            'All': union
        }
    return res

