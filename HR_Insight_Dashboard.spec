# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_all

datas = [('pj7.py', '.')]
binaries = []
hiddenimports = [
    # xgboost
    'xgboost', 'xgboost.core', 'xgboost.sklearn',
    # sklearn internals
    'sklearn.utils._typedefs', 'sklearn.utils._heap', 'sklearn.utils._sorting',
    'sklearn.utils._vector_sentinel', 'sklearn.utils._weight_vector',
    # scipy
    'scipy.special._cdflib', 'scipy.special._ufuncs', 'scipy.special.cython_special',
    'scipy.special._specfun', 'scipy._lib.messagestream',
    # numpy
    'numpy', 'numpy.core', 'numpy.random', 'numpy.linalg',
    'numpy.core._multiarray_umath', 'numpy.core._multiarray_tests',
    # shap - 전체 서브모듈 명시
    'shap', 'shap._explanation', 'shap.explainers', 'shap.explainers._tree',
    'shap.explainers._gpu_tree', 'shap.explainers._deep', 'shap.explainers._gradient',
    'shap.explainers._kernel', 'shap.explainers._linear', 'shap.explainers._exact',
    'shap.explainers._permutation', 'shap.explainers._partition',
    'shap.explainers._sampling', 'shap.explainers._additive',
    'shap.maskers', 'shap.maskers._tabular', 'shap.maskers._text',
    'shap.maskers._image', 'shap.maskers._fixed_composite', 'shap.maskers._composite',
    'shap.plots', 'shap.plots._bar', 'shap.plots._beeswarm', 'shap.plots._scatter',
    'shap.plots._waterfall', 'shap.plots._force', 'shap.plots._decision',
    'shap.utils', 'shap.utils._clustering', 'shap.utils._legacy',
    'shap.utils._masked_model', 'shap.utils._explain',
    # openpyxl
    'openpyxl', 'openpyxl.styles', 'openpyxl.styles.stylesheet',
    'openpyxl.writer.excel', 'openpyxl.reader.excel',
    'openpyxl.workbook', 'openpyxl.worksheet',
    # xlrd
    'xlrd',
    # pandas
    'pandas', 'pandas._libs.tslibs.timedeltas', 'pandas._libs.tslibs.np_datetime',
    # threading / webbrowser
    'threading', 'webbrowser',
]

tmp_ret = collect_all('streamlit')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('plotly')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('sklearn')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('shap')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

datas += collect_data_files('xgboost')

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # UI / 시각화 (불필요)
        'tkinter', 'matplotlib', 'IPython', 'jupyter', 'notebook',
        'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'wx',
        'panel', 'bokeh', 'holoviews', 'hvplot',
        # 딥러닝 (불필요)
        'torch', 'torchvision', 'torchaudio',
        'tensorflow', 'keras', 'tf2onnx',
        'onnxruntime', 'onnx',
        'transformers', 'tokenizers', 'accelerate', 'diffusers',
        # 수치계산 옵셔널 (불필요)
        'numba', 'llvmlite', 'cupy',
        # 클라우드 SDK (불필요)
        'boto3', 'botocore', 'aiobotocore', 's3transfer',
        'google', 'googleapiclient', 'google.cloud',
        'azure',
        # 미디어 (불필요)
        'av', 'cv2', 'imageio', 'PIL',
        # 기타 대형 패키지 (불필요)
        'dask', 'distributed', 'ray',
        'sqlalchemy', 'alembic',
        'grpc', 'grpcio',
        'pyspark', 'pyarrow.gandiva',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HR_Insight_Dashboard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='HR_Insight_Dashboard',
)
app = BUNDLE(
    coll,
    name='HR_Insight_Dashboard.app',
    icon=None,
    bundle_identifier=None,
)
