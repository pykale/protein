def register_builtin_components():
    from .modalities.small_molecule import processors as _smp, encoders as _sme
    from .modalities.protein_sequence import processors as _psp, encoders as _pse
    from .modalities.protein_structure import processors as _pstp, encoders as _pste
    from .fusion import concat as _concat, bilinear_attention as _ba, cross_attention as _ca
    from .conditioners import structure_conditioned_denoising as _scd
    from .heads import classification as _cl, regression as _reg, diffusion_sequence_decoder as _dsd
    from .runners import predict_runner as _pr, diffusion_generate_runner as _dgr
    from .tasks.drug_target_interaction import metrics as _dtim, interpreters as _dtii
    from .tasks.inverse_folding import metrics as _ifm, interpreters as _ifi
    from .tasks.drug_target_interaction import datasets as _dtid
    from .tasks.inverse_folding import datasets as _ifd
    from .examples import drugban_dti as _drugban_card
    from .examples import mapdiff_inverse_folding as _mapdiff_card
    return True
register_builtin_components()
