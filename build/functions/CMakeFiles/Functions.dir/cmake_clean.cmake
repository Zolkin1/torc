file(REMOVE_RECURSE
  "libFunctions.pdb"
  "libFunctions.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/Functions.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
