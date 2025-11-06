#!/bin/awk -f

function assert(condition, string) {
    if (!condition) {
        printf("%s:%d: assertion failed: %s\n", FILENAME, FNR, string) > "/dev/stderr"
        _assert_exit = 1
        exit 1
    }
}

/Dataset size:/ {
   assert(dataset_size == "", "Multiple occurencies of dataset size!")
   dataset_size = $NF
}

/Number of samples:/ {
   assert(max_samples == "", "Multiple occurencies of max no. samples!")
   max_samples = $NF
   assert(max_samples < dataset_size, "max_samples < dataset_size : " max_samples " < " dataset_size)
}

/Number of variables:/ {
   assert(variables_cnt == "", "Multiple occurencies of no. variables!")
   variables_cnt = $NF
}

/expected output:/ {
   expected = $NF
}

/computed output:/ {
   computed = $NF
   assert(expected != "" && computed != "")
   total_cnt++
   if (computed == expected) correct_cnt++
}

/#features:/ {
   split($NF, nums, "/")
   sum_size += nums[1]
   div_sum_size += nums[2]
}

/#fixed features:/ {
   split($NF, nums, "/")
   sum_fixed += nums[1]
   div_sum_fixed += nums[2]
}

/#terms:/ {
   sum_termsize += $NF
   cnt_termsize++
}

/#checks:/ {
   sum_checks += $NF
   cnt_checks++
}

/<timeout>/ {
   timeouts_cnt++
}

/relVolume\*:/ {
   split($NF, nums, "%")
   sum_volume += nums[1]
   cnt_volume++
}

END {
   if (_assert_exit)
      exit 1

   if (total_cnt > 0) {
      print("Total: " total_cnt)
      if (max_samples == "") {
         assert(total_cnt == dataset_size, "total_cnt == dataset_size: " total_cnt " == " dataset_size)
      } else {
         assert(total_cnt == max_samples, "total_cnt == max_samples: " total_cnt " == " max_samples)
      }

      if (cnt_checks > 0) {
         assert(total_cnt == 0 || total_cnt == cnt_checks, "total_cnt == cnt_checks: " total_cnt " == " cnt_checks)
      }

      if (timeouts_cnt == "") timeouts_cnt = 0
      completed_cnt = total_cnt - timeouts_cnt

      if (cnt_volume > 0) {
         assert(completed_cnt == 0 || completed_cnt == cnt_volume, "completed_cnt == cnt_volume: " completed_cnt " == " cnt_volume)
      }
   }
   if (variables_cnt > 0) {
      print("Features: " variables_cnt)
   }
   if (total_cnt > 0) {
      print("Timeouts: " timeouts_cnt)
      # printf("avg #timeouts: %.1f%%\n", (timeouts_cnt/total_cnt)*100)
      printf("avg #completed: %.1f%%\n", (completed_cnt/total_cnt)*100)
      printf("avg #correct classifications (incl. timeouted): %.1f%%\n", (correct_cnt/total_cnt)*100)
   }
   if (div_sum_size > 0) {
      printf("avg #any features: %.1f%%\n", (sum_size/div_sum_size)*100)
      if (div_sum_fixed > 0) printf("avg #fixed features: %.1f%%\n", (sum_fixed/div_sum_fixed)*100)
      else printf("avg #fixed features: %.1f%%\n", (sum_size/div_sum_size)*100)
   }
   if (cnt_termsize > 0) {
      printf("avg #terms: %.1f\n", (sum_termsize/cnt_termsize))
   }
   if (cnt_volume > 0) {
      printf("avg relVolume*: %.2f%%\n", sum_volume/cnt_volume)
   }
   if (cnt_checks > 0) {
      printf("avg #checks: %.1f\n", (sum_checks/cnt_checks))
   }
}
