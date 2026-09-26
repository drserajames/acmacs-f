# Golden fixtures for af.chart.control, produced by the R package hicontrol itself.
#
#   Rscript make_fixtures.R <hicontrol source dir> <output .json>
#
# hicontrol (Sarah James and Yi Dong, GPL-3) is sourced directly from its R/ directory, so no
# install is needed. Every case records the inputs and R's exact result: NULL vs a zero-length
# vector, matrix shape, NA entries, and any error, so the Python port can be checked for
# equivalence case by case. Inputs are synthetic: the package's own testthat cases, plus
# seeded random series built to exercise ties, missing readings, censoring and each rule.

args <- commandArgs(trailingOnly = TRUE)
src <- args[[1]]
out <- args[[2]]
for (f in c("utils.R", "rules.R", "hi_rules.R")) source(file.path(src, "R", f))
commit <- trimws(readLines(file.path(src, "COMMIT"), warn = FALSE)[1])

enc <- function(x) {
  if (is.null(x)) return(list(kind = "null"))
  if (is.matrix(x)) {
    return(list(kind = "matrix", nrow = nrow(x), ncol = ncol(x),
                rows = lapply(seq_len(nrow(x)), function(i) I(as.numeric(unname(x[i, ]))))))
  }
  if (is.list(x)) return(list(kind = "list", items = lapply(x, enc)))
  list(kind = "vector", values = I(as.numeric(unname(x))))
}

run <- function(expr) {
  tryCatch(enc(suppressWarnings(eval(expr))), error = function(e) list(kind = "error", message = conditionMessage(e)))
}

cases <- list()
# `series`, not `dat`: rule_diff's `d = 3` would partially match an argument called `dat`.
add <- function(fn, series, ...) {
  params <- list(...)
  stopifnot(!is.null(names(params)), all(names(params) != ""))
  call <- as.call(c(as.name(fn), list(dat = series), params))
  cases[[length(cases) + 1]] <<- list(fn = fn, dat = I(as.numeric(series)), params = params, result = run(call))
}

# the package's own test inputs --------------------------------------------------------
add("rule_xyz", c(5, 5, 10, 5, 5), centre = 5, threshold = 1, x = 1, y = 1, z = 3)
add("rule_xyz", c(5, 5, 0, 5, 5), centre = 5, threshold = 1, x = 1, y = 1, z = 3)
add("rule_xyz", c(5, 5, 7, 5, 5), centre = 5, threshold = 1, x = 1, y = 1, z = 3)
add("rule_xyz", c(5, 5), centre = 5, threshold = 1, x = 1, y = 3, z = 2)
add("rule_xyz", c(8, 8, 5, 5, 5), centre = 5, threshold = 1, x = 2, y = 3, z = 2)
add("rule_xyz", c(8, 5, 5, 5, 5), centre = 5, threshold = 1, x = 2, y = 3, z = 2)
add("rule_xyz", c(5, 10, 5, 0, 5), centre = 5, threshold = 1, x = 1, y = 1, z = 3)
for (d in list(c(1, 2, 3, 4, 5), c(5, 4, 3, 2, 1), c(1, 3, 2, 4, 3, 5), c(1, 2, 3), c(1, 2, 3, 2, 1),
               c(5, 5, 1, 2, 3, 4, 5, 5), c(9, 7, 5, 3, 1, 4, 7))) add("rule_trend", d, n = 4)
add("rule_noC", c(3, 4, 3, 4, 3, 4, 0), centre = 0, threshold = 1, n = 6)
add("rule_noC", c(0, 0.5, -0.5, 0), centre = 0, threshold = 1, n = 3)
add("rule_noC", c(3, 4, 0, 0, 0, 0), centre = 0, threshold = 1, n = 4)
add("rule_noC", c(1, 1, 1, 1, 1, 1), centre = 0, threshold = 1, n = 5)
add("rule_noC", c(3, 4, 3), centre = 0, threshold = 1, n = 5)
add("rule_noC", c(3, 4, 3, 4, 0, 3, 4, 3, 4), centre = 0, threshold = 1, n = 3)
add("rule_noC", c(1, 1, 1, 1, 1, 0), centre = 0, threshold = 1, n = 5)
add("rule_onlyC", c(0.5, -0.5, 0.3, -0.3, 0.5, 5), centre = 0, threshold = 1, n = 5)
add("rule_onlyC", c(0.5, -0.5, 5, 0.5, -0.5), centre = 0, threshold = 1, n = 3)
add("rule_onlyC", c(5, 0.5, -0.5, 0.3, -0.3, 0.5, 5), centre = 0, threshold = 1, n = 5)
add("rule_onlyC", c(1, -1, 1, -1, 1), centre = 0, threshold = 1, n = 5)
add("rule_onlyC", c(0.5, -0.5, 0.3), centre = 0, threshold = 1, n = 5)
for (d in list(c(1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1), c(1, 5, 1, 5, 1, 5, 1, 5, 1, 5), c(1, 5, 1, 5, 1, 5, 1, 5, 1),
               1:10, c(1, 5, 1, 5, 1), c(3, 3, 3, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 3, 3), c(1, 4, 2, 5, 1, 6, 2, 4, 1, 5, 2)))
  add("rule_alt", d, n = 10)
for (d in list(c(1, 2, 2, 2, 2, 2, 3), c(1, 2, 2, 2, 2, 2), c(1, 2, 2, 2, 2, 3), 1:10, c(2, 2, 2), c(3, 3, 3, 3, 3, 1)))
  add("rule_nodiff", d, n = 5)
for (d in list(c(1, 2, 5, 6, 7), c(1, 2, 3, 4, 5), c(1, 6, 1, 6, 1), c(1, 4), c(1, 3.9, 1), c(5, 1, 5)))
  add("rule_diff", d, d = 3)
add("rule_noCrelax", c(3, 1, 3, 1, 3, 1, 0), centre = 0, threshold = 1, n = 6)
add("rule_noCrelax", c(1, 1, 1, 1, 1, 0), centre = 0, threshold = 1, n = 5)
add("rule_noCrelax", c(3, 4, 0, 0, 0, 0), centre = 0, threshold = 1, n = 5)
add("rule_noCrelax", c(3, 4, 3), centre = 0, threshold = 1, n = 5)
for (d in list(c(2, 4, 6, 8, 6, 4, 2, 4, 6, 8), c(0, 0, 0, 10, 0, 0, 0, 0, 0, 0), rep(0, 26),
               c(0, 0, 1, 2, 3, 4, 5, 0, 0, 0), c(0, 0, 0, 5, 0, 0, 0, 0, 0, 0))) {
  add("hi_rules", d, centre = 4, threshold = 1)
  add("hi_rules2", d, centre = 0, threshold = 1)
}
add("hi_rules", c(0, 0.1, -0.1, 0, 0.1, -0.1, 0, 0.1, -0.1, 0), centre = 0, threshold = 5)

# generated series ----------------------------------------------------------------------
set.seed(20260926)
gen <- function(len, kind) {
  switch(kind,
    discrete = sample(-1:8, len, replace = TRUE, prob = c(1, 1, 2, 3, 5, 8, 5, 3, 2, 1)),
    tight = 4 + sample(c(-1, 0, 0, 0, 1), len, replace = TRUE),
    normal = round(rnorm(len, 4, 1.3), 3),
    na = { v <- sample(0:8, len, replace = TRUE); v[runif(len) < 0.2] <- NA; v },
    trend = cumsum(sample(c(-1, 1, 1, 1), len, replace = TRUE)),
    alt = rep(c(3, 5), length.out = len) + sample(c(0, 0, 0, 1), len, replace = TRUE),
    flat = c(rep(4, max(0, len - 3)), sample(0:8, min(3, len), replace = TRUE)))
}
kinds <- c("discrete", "tight", "normal", "na", "trend", "alt", "flat")
for (rep in 1:40) for (kind in kinds) {
  len <- sample(c(0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 15, 25, 26, 30, 40), 1)
  d <- gen(len, kind)
  centre <- if (all(is.na(d))) 0 else median(d, na.rm = TRUE)
  threshold <- sample(c(1, 1, 1, 0.5, 1.5), 1)
  add("rule_xyz", d, centre = centre, threshold = threshold, x = 1, y = 1, z = 3 - 1e-5)
  add("rule_xyz", d, centre = centre, threshold = threshold, x = 2, y = 3, z = 2 - 1e-5)
  add("rule_xyz", d, centre = centre, threshold = threshold, x = 8, y = 8, z = 0)
  add("rule_noCrelax", d, centre = centre, threshold = threshold, n = 9)
  add("rule_noC", d, centre = centre, threshold = threshold, n = sample(3:8, 1))
  add("rule_onlyC", d, centre = centre, threshold = threshold, n = sample(3:8, 1))
  add("rule_nodiff", d, n = sample(c(3, 5, 25), 1))
  add("rule_alt", d, n = sample(c(4, 6, 10), 1))
  add("rule_trend", d, n = sample(c(3, 4, 5), 1))
  add("rule_diff", d, d = 3)
  add("hi_rules", d, centre = centre, threshold = threshold)
  add("hi_rules2", d, centre = centre, threshold = threshold)
}

titres <- list(c("10", "20", "40", "80", "160", "320"), c("<10", "<20", "<40"), c("40", "80", "<10", "160"),
               c("<10", "<10", "<20"), c("40"), c("*", "40", ">2560", "<10", "1280", "."), c("15", "21", "1467"))
logs <- lapply(titres, function(t) list(titres = I(t), result = enc(suppressWarnings(log_num(t)))))

writeLines(jsonlite::toJSON(list(source = "hicontrol", commit = commit, cases = cases, log_num = logs),
                            auto_unbox = TRUE, null = "null", na = "null", digits = I(17)), out)
cat(length(cases), "rule cases,", length(logs), "log_num cases ->", out, "\n")
