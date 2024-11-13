library(tidymodels)
library(vroom)
library(embed)
library(glmnet)
library(naivebayes)
library(discrim)
library(themis)

train <- vroom("train.csv")
test<- vroom("test.csv")

nb_model <- naive_Bayes(Laplace = tune(),
                          smoothness = tune()
) %>%
  set_engine("naivebayes") %>%
  set_mode("classification")

nb_recipe <- recipe(type ~ ., data = train) %>%
  step_mutate_at(color, fn = factor) %>%
  step_normalize(all_numeric(), -all_outcomes()) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>%
  step_smote(all_outcomes(), neighbors = 3)

nb_wf <- workflow() %>%
  add_model(nb_model) %>%
  add_recipe(nb_recipe)

nb_grid <- grid_regular(Laplace(), smoothness(),
                           levels = 20
)

cv_folds <- vfold_cv(train, v = 10, repeats = 1)

tuned_results <- nb_wf |>
  tune_grid(resamples = cv_folds,
            grid = nb_grid,
            metrics = metric_set(roc_auc))

best_params <- tuned_results |>
  select_best(metric = "roc_auc")

best_params

nb_final_wf <- nb_wf %>%
  finalize_workflow(best_params) %>%
  fit(train)

predictions <- predict(nb_final_wf,
                       new_data = test,
                       type = "class") 

kaggle_submission <- predictions %>%
  bind_cols(., test) %>%
  dplyr::select(id,.pred_class) %>%
  rename(type = .pred_class)

vroom_write(kaggle_submission, "NaiveBayespreds.csv", delim = ",")