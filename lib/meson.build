project('tsls', 'c', default_options: ['c_std=c11'])

# The tskit C library is downloaded with the wrap file in subprojects/
# (see subprojects/README).
tskit_proj = subproject('tskit')
tskit_dep = tskit_proj.get_variable('tskit_dep')

cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required : false)
cunit_dep = dependency('cunit')

extra_c_args = [
    '-Wall', '-Wextra', '-Werror', '-Wpedantic', '-W',
    '-Wmissing-prototypes',  '-Wstrict-prototypes',
    '-Wconversion', '-Wshadow', '-Wpointer-arith', '-Wcast-align',
    '-Wcast-qual', '-Wwrite-strings', '-Wnested-externs',
    '-fshort-enums', '-fno-common']

tsls_sources = ['hmm.c', 'err.c', 'blockalloc.c']

tsls_lib = static_library('tsls',
    sources: tsls_sources, dependencies: [m_dep, tskit_dep],
    c_args: extra_c_args)


unit_tests = executable('tests',
    sources: ['tests/tests.c'],
    link_with: [tsls_lib], dependencies:[cunit_dep, tskit_dep])
test('Unit tests', unit_tests)
