
from ml.model.common import CLF_IMPL, CLF_NAME, resolve_clf
from ml.model.saveable import SaveableMixin


def prep_for_save(res):
    for fold_res in res:
        for clf_res in fold_res:
            clf = clf_res['model'][CLF_IMPL]
            clf = resolve_clf(clf)
            if isinstance(clf, SaveableMixin):
                clf.prepare_for_save()
    return res


def restore_from_save(res):
    for fold_res in res:
        for clf_res in fold_res:
            clf = clf_res['model'][CLF_IMPL]
            clf = resolve_clf(clf)
            if isinstance(clf, SaveableMixin):
                clf.restore_from_save()
    return res
