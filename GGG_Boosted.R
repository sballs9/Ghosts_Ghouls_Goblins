library(tidymodels)
library(vroom)
library(bonsai)
library(lightgbm)

train <- vroom("train.csv")
test<- vroom("test.csv")

train <- train %>%
  mutate(across(where(is.character), as.factor))

boost_model <- boost_tree(tree_depth = 1,
                          trees = tune(),
                          learn_rate = 0.1
                          ) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

boost_recipe <- recipe(type ~ ., data = train) %>%
  update_role(id, new_role="id") %>%
  step_dummy(color) %>%
  step_zv()

boost_wf <- workflow() %>%
  add_model(boost_model) %>%
  add_recipe(boost_recipe)

boost_grid <- grid_regular(trees(range = c(50, 500)),
  levels = 5
)

cv_folds <- vfold_cv(train, v = 10, repeats = 1)

tuned_results <- boost_wf |>
  tune_grid(resamples = cv_folds,
            grid = boost_grid,
            metrics = metric_set(accuracy))

best_params <- tuned_results |>
  select_best(metric = "accuracy")

best_params

boost_final_wf <- boost_wf %>%
  finalize_workflow(best_params) %>%
  fit(train)

predictions <- predict(boost_final_wf,
                       new_data = test,
                       type = "class") 

kaggle_submission <- predictions %>%
  bind_cols(., test) %>%
  dplyr::select(id,.pred_class) %>%
  rename(type = .pred_class)

vroom_write(kaggle_submission, "Boostedpreds.csv", delim = ",")