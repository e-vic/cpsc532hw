(ns daphne.linalg-test
  (:require [clojure.test :refer [deftest testing is]]
            [daphne.linalg :refer [foppl-linalg]]
            [daphne.core :refer [code->graph]]))


(deftest convolution-test
  (testing "Test convolution function."
    ;; pytorch conv2d validated
    [[[2.202277681399303 2.8281133363421826]
      [2.230589302485512 1.7424331895209988]]
     [[1.4150513754814487 1.5241574765307004]
      [1.3686725762513385 1.2358384037379724]]]
    (is (= [[[2.201277681399303 2.8271133363421828]
             [2.229589302485512 1.741433189520999]]
            [[1.4130513754814487 1.5221574765307004]
             [1.3666725762513385 1.2338384037379724]]]
           (last
            (code->graph
             (concat foppl-linalg
                     '((let [w1 [[[[0.8097745873461849, 0.9027974133677656],
                                   [0.9937591748266679, 0.6899363139420105]],
                                  [[0.09797226233334677, 0.02146334941825967],
                                   [0.13535254818829612, 0.5766735975714063]]],
                                 [[[0.21673669826346842, 0.4318042477853944],
                                   [0.6986981163149986, 0.10796200682913093]],
                                  [[0.4354448007571432, 0.5948937288685611],
                                   [0.10808562241514497, 0.10665190515628087]]]]

                             b1 [0.001, 0.002]
                             x  [[[0.4287027640286398, 0.5976520946846761, 0.8412232029516122, 0.843736605231717]
                                  [0.564107482170524, 0.8311282657110909, 0.6649508631095007, 0.18790346905566147]
                                  [0.9061368614009218, 0.4177246719504131, 0.7508225882288696, 0.24750179094034197]
                                  [0.5116059526254529, 0.06562150286776547, 0.4562421518588141, 0.24021920534487728]],
                                 [[0.05876861913939957, 0.8392780098655948, 0.2159165110183996, 0.37296811984042133]
                                  [0.29292562830682534, 0.20312029690945455, 0.8465110915686057, 0.7803549289269531]
                                  [0.14643684731359985, 0.6894799714557894, 0.6801510765311485, 0.3989642366533286]
                                  [0.6300020585310742, 0.7813718161440676, 0.5554317792622283, 0.24360960738915194]]]
                             ]
                         (conv2d x w1 b1 2)
                         ))
                     )))))))
