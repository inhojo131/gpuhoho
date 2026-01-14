common_src += files(['CGeometry.cpp',
                     'CPhysicalGeometry.cpp',
                     'CPhysicalGeometryFEM.cpp',
                     'CMultiGridGeometry.cpp',
                     'CDummyGeometry.cpp',
                     'CMultiGridQueue.cpp'])

# GPU geometry buffer (only when CUDA is enabled)
if get_option('enable-cuda')
  common_src += files(['GeometryGPU.cpp'])
endif
