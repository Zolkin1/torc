file(REMOVE_RECURSE
  "libConstraints.pdb"
  "libConstraints.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/Constraints.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
